from typing import Literal, Iterable, Optional, Type, Dict, List
from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
import math
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, IterableDataset
import torch
import supervision as sv

from trackers.ball_tracker.models import TrackNet, InpaintNet
from trackers.ball_tracker.dataset import BallTrajectoryDataset
from trackers.ball_tracker.iterable import BallTrajectoryIterable
from trackers.ball_tracker.predict import predict, predict_modified, predict_bbox_and_confidence
from trackers.tracker import Object, Tracker, NoPredictSample



def get_model(
    model_name: Literal["TrackNet", "InpaintNet"], 
    seq_len: int = None, 
    bg_mode: Literal["", "subtract", "subtract_concat", "concat"] = None,
) -> torch.nn.Module:
    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            model = TrackNet(in_dim=seq_len*4, out_dim=seq_len)
        elif bg_mode == 'concat':
            model = TrackNet(in_dim=(seq_len+1)*3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len*3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name.')
    
    return model


def get_ensemble_weight(
    seq_len: int, 
    eval_mode: Literal["average", "weight"],
) -> torch.Tensor:
    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')
    
    return weight


def generate_inpaint_mask(pred_dict: dict, th_h: float=30) -> list:
    y = np.array(pred_dict['y'])
    vis_pred = np.array(pred_dict['visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0
    j = 0 
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred)-1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred)-1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            pass
        i = j
    
    return inpaint_mask.tolist()


class Ball(Object):
    def __init__(
        self, 
        frame: int, 
        xy: tuple[float, float], 
        visibility: Literal[0,1],
        projection: Optional[tuple[int, int]] = None                    
    ):
        super().__init__()

        self.frame = frame
        self.xy = xy
        self.visibility = visibility
        self.projection = projection

    @classmethod
    def from_json(cls, x: dict):
        return cls(**x)

    def serialize(self) -> dict:
        return {
            "frame": self.frame,
            "xy": self.xy,
            "visibility": self.visibility,
            "projection": self.projection,
        }
    
    def asint(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.xy)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw ball detection in a given frame
        """

        cv2.circle(
            frame,
            self.asint(),
            6,
            (0, 255, 0),
            -1,
        )

        return frame
    
    def draw_projection(self, frame: np.ndarray) -> np.ndarray:
        
        cv2.circle(
            frame,
            self.projection,
            6,
            (255, 255, 0),
            -1,
        )

        return frame


class TRBall(Tracker):
    EVAL_MODE: str = "weight"
    TRAJECTORY_LENGTH: int = 8
    
    HEIGHT: int = 288
    WIDTH: int = 512
    SIGMA: float = 2.5
    IMG_FORMAT = 'png'
    
    def __init__(
        self, 
        tracking_model_path: str,
        inpainting_model_path: str,
        batch_size: int,
        median_max_sample_num: int = 1800, 
        median: Optional[np.ndarray] = None,
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ):
        super().__init__(
            load_path=load_path,
            save_path=save_path,
        )

        self.DELTA_T: float = 1 / math.sqrt(self.HEIGHT**2 + self.WIDTH**2)
        self.COOR_TH = self.DELTA_T * 50

        tracknet_ckpt = torch.load(tracking_model_path, map_location='cpu', weights_only=False)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']

        assert self.tracknet_seq_len == self.TRAJECTORY_LENGTH

        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']

        self.tracknet = get_model(
            "TrackNet", 
            self.tracknet_seq_len,
            self.bg_mode,
        )
        self.tracknet.load_state_dict(tracknet_ckpt['model'])
        self.tracknet.eval()

        if inpainting_model_path:
            inpaintnet_ckpt = torch.load(inpainting_model_path, map_location='cpu', weights_only=False)
            self.inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
            self.inpaintnet = get_model('InpaintNet')
            self.inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        else:
            self.inpaintnet = None

        self.batch_size = batch_size
        self.median_max_sample_num = median_max_sample_num
        self.median = median
    
    def video_info_post_init(self, video_info: sv.VideoInfo) -> "TRBall":
        self.video_info = video_info
        return self
    
    def object(self) -> Type[Object]:
        return Ball
    
    def draw_kwargs(self) -> dict:
        return {}
    
    def __str__(self) -> str:
        return "ball_tracker"
    
    def restart(self) -> None:
        self.results.restart()

    def processor(self, frame: np.ndarray):
        pass
    
    def draw_traj(self, img, traj, radius=3, color='red') -> np.ndarray:
        img = Image.fromarray(img)
        
        for i in range(len(traj)):
            if traj[i] is not None:
                draw_x = traj[i][0]
                draw_y = traj[i][1]
                bbox =  (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
                draw = ImageDraw.Draw(img)
                draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
                del draw

        return np.array(img)
    
    def draw_multiple_frames(
        self,
        frames: list[np.ndarray],
        ball_detections: list[Ball],
        traj_len=8
    ):

        pred_queue = deque()
        
        output_frames = []
        for frame, ball_detection in zip(frames, ball_detections):
            if len(pred_queue) >= traj_len:
                pred_queue.pop()
        
            pred_queue.appendleft(
                list(ball_detection.xy)
            ) if ball_detection.visibility else pred_queue.appendleft(None)
            output_frames.append(self.draw_traj(frame, pred_queue, color='yellow'))

        return output_frames
    
    def modify_pred_dict(self, pred_dict: dict):

        mapping = {
            "X": "x",
            "Y": "y",
            "Visibility": "visibility",
            "Inpaint_Mask": "inpaint_mask",
            "Img_scaler": "img_scaler",
            "Img_shape": "img_shape",
        }

        return {
            k: pred_dict[v]
            for k, v in mapping.items()
        }
    
    def to(self, device: str) -> None:
        self.tracknet.to(device)
        if self.inpaintnet is not None:
            self.inpaintnet.to(device)

    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs):
        raise NoPredictSample()

    def predict_frames(
        self,
        frame_generator: Iterable[np.ndarray],
        total_frames: int,
    ) -> list[Ball]:

        w_scaler, h_scaler = (
            self.video_info.width / self.WIDTH, 
            self.video_info.height / self.HEIGHT,
        )

        img_scaler = (w_scaler, h_scaler)

        tracknet_pred_dict = {
            'frame':[], 
            'x':[], 
            'y':[], 
            'visibility':[], 
            'inpaint_mask': [],
            'img_scaler': img_scaler, 
            'img_shape': (self.video_info.width, self.video_info.height),
        }

        seq_len = self.tracknet_seq_len

        iterable = BallTrajectoryIterable(
            seq_len=seq_len,
            sliding_step=1,
            data_mode="heatmap",
            bg_mode="concat",
            frame_generator=frame_generator,
            HEIGHT=self.HEIGHT,
            WIDTH=self.WIDTH,
            SIGMA=2.5,
            IMG_FORMAT="png",
            median=self.median,
            median_range=self.median_max_sample_num,
        )

        data_loader = DataLoader(
            iterable,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        video_len = total_frames
        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        sample_indices = torch.arange(seq_len)
        frame_indices = torch.arange(seq_len-1, -1, -1)
        y_pred_buffer = torch.zeros(
            (
                buffer_size, 
                seq_len, 
                self.HEIGHT, 
                self.WIDTH
            ), 
            dtype=torch.float32,
        )

        weight = get_ensemble_weight(seq_len, self.EVAL_MODE)

        for x in tqdm(data_loader):
            x = x.float().to(self.DEVICE)

            batch_size = x.shape[0]
            assert seq_len*3 + 3 == x.shape[1] 

            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat(
                (y_pred_buffer, y_pred), 
                dim=0,
            )

            ensemble_y_pred = torch.empty(
                (0, 1, self.HEIGHT, self.WIDTH), 
                dtype=torch.float32,
            )

            for sample_i in range(batch_size):
                if sample_count < buffer_size:
                    y_pred = y_pred_buffer[
                        sample_indices + sample_i,
                        frame_indices,
                    ].sum(0) / (sample_count + 1)
                else:
                    y_pred = (
                        y_pred_buffer[
                            sample_indices + sample_i,
                            frame_indices
                        ] * weight[:, None, None]
                    ).sum(0)

                ensemble_y_pred = torch.cat(
                    (
                        ensemble_y_pred, 
                        y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH),
                    ),
                    dim=0,
                )
                sample_count += 1

                if sample_count == num_sample:
                    y_zero_pad = torch.zeros(
                        (buffer_size, seq_len, self.HEIGHT, self.WIDTH),
                        dtype=torch.float32,
                    )
                    y_pred_buffer = torch.cat(
                        (y_pred_buffer, y_zero_pad),
                        dim=0,
                    )
                    print(seq_len)
                    for frame_i in range(1, seq_len):
                        y_pred = y_pred_buffer[
                            sample_indices + sample_i + frame_i,
                            frame_indices
                        ].sum(0) / (seq_len - frame_i)

                        ensemble_y_pred = torch.cat(
                            (
                                ensemble_y_pred, 
                                y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH),
                            ),
                            dim=0,
                        )

            tmp_pred = predict_modified(
                y_pred=ensemble_y_pred,
                img_scaler=img_scaler,
                WIDTH=self.WIDTH,
                HEIGHT=self.HEIGHT,
            )

            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            y_pred_buffer = y_pred_buffer[-buffer_size:]

        if self.inpaintnet is not None:
            self.inpaintnet.eval()
            seq_len = self.inpaintnet_seq_len
            tracknet_pred_dict["inpaint_mask"] = generate_inpaint_mask(
                tracknet_pred_dict, th_h=self.video_info.height*0.05,
            )
            inpaint_pred_dict = {
                'Frame':[], 
                'X':[], 
                'Y':[], 
                'Visibility':[],
            }

            dataset = BallTrajectoryDataset(
                seq_len=seq_len, 
                sliding_step=1, 
                data_mode='coordinate', 
                pred_dict=self.modify_pred_dict(tracknet_pred_dict),
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                SIGMA=self.SIGMA,
                IMG_FORMAT=self.IMG_FORMAT,
            )
            data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False,
            )

            weight = get_ensemble_weight(seq_len, self.EVAL_MODE)
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            sample_indices = torch.arange(seq_len) 
            frame_indices = torch.arange(seq_len-1, -1, -1) 
            coor_inpaint_buffer = torch.zeros(
                (buffer_size, seq_len, 2), 
                dtype=torch.float32,
            )

            for (i, coor_pred, inpaint_mask) in tqdm(data_loader):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                batch_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = self.inpaintnet(
                        coor_pred.cuda(), 
                        inpaint_mask.cuda(),
                    ).detach().cpu()
                    
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                th_mask = (
                    (
                        (coor_inpaint[:, :, 0] < self.COOR_TH) 
                        &
                        (coor_inpaint[:, :, 1] < self.COOR_TH)
                    )
                )
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat(
                    (coor_inpaint_buffer, coor_inpaint),
                    dim=0,
                )
                ensemble_i = torch.empty(
                    (0, 1, 2), 
                    dtype=torch.float32,
                )
                ensemble_coor_inpaint = torch.empty(
                    (0, 1, 2), 
                    dtype=torch.float32,
                )
                
                for sample_i in range(batch_size):
                    if sample_count < buffer_size:
                        coor_inpaint = coor_inpaint_buffer[
                            sample_indices + sample_i, 
                            frame_indices,
                        ].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        coor_inpaint = (
                            coor_inpaint_buffer[
                                sample_indices + sample_i, 
                                frame_indices,
                            ] * weight[:, None]
                        ).sum(0)
                    
                    ensemble_i = torch.cat(
                        (ensemble_i, i[sample_i][0].view(1, 1, 2)), 
                        dim=0,
                    )
                    ensemble_coor_inpaint = torch.cat(
                        (ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), 
                        dim=0,
                    )
                    sample_count += 1

                    if sample_count == num_sample:
                        coor_zero_pad = torch.zeros(
                            (buffer_size, seq_len, 2), 
                            dtype=torch.float32,
                        )
                        coor_inpaint_buffer = torch.cat(
                            (coor_inpaint_buffer, coor_zero_pad), 
                            dim=0,
                        )
                        
                        for frame_i in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[
                                sample_indices + sample_i + frame_i, 
                                frame_indices
                            ].sum(0)
                            coor_inpaint /= (seq_len - frame_i)
                            ensemble_i = torch.cat(
                                (ensemble_i, i[-1][frame_i].view(1, 1, 2)), 
                                dim=0,
                            )
                            ensemble_coor_inpaint = torch.cat(
                                (ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), 
                                dim=0,
                            )
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < self.COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < self.COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                tmp_pred = predict(
                    ensemble_i, 
                    c_pred=ensemble_coor_inpaint,
                    img_scaler=img_scaler, 
                    WIDTH=self.WIDTH, 
                    HEIGHT=self.HEIGHT,
                )

                {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        pred_dict = inpaint_pred_dict if self.inpaintnet is not None else tracknet_pred_dict
        
        ball_detections = []
        for frame_counter in range(video_len):
            if frame_counter in pred_dict["Frame"]:
                i = pred_dict["Frame"].index(frame_counter)
                ball_detections.append(
                    Ball(
                        frame=frame_counter,
                        xy=(pred_dict["X"][i], pred_dict["Y"][i]),
                        visibility=pred_dict["Visibility"][i]
                    )
                )
            else:
                print(f"{self.__str__()}: missing detection frame {frame_counter}")
                ball_detections.append(
                    Ball(
                        frame=frame_counter,
                        xy=(0.0, 0.0),
                        visibility=0,
                    )
                )
        return ball_detections

    def predict_frames_eval(
        self,
        frame_generator: Iterable[np.ndarray],
        total_frames: int,
        heatmap_threshold: float = 0.5,
    ) -> List[Dict]:
        if not hasattr(self, 'video_info') or self.video_info is None:
             raise ValueError("BallTracker must have video_info set via video_info_post_init before prediction.")

        w_scaler, h_scaler = (self.video_info.width / self.WIDTH, self.video_info.height / self.HEIGHT,)
        img_scaler = (w_scaler, h_scaler)
        tracknet_seq_len = self.tracknet_seq_len
        
        print(f"[DEBUG Params] bg_mode: {self.bg_mode}, EVAL_MODE: {self.EVAL_MODE}, heatmap_thresh: {heatmap_threshold}")

        iterable_tn = BallTrajectoryIterable(
            seq_len=tracknet_seq_len,
            sliding_step=1,
            data_mode="heatmap",
            bg_mode=self.bg_mode,
            frame_generator=frame_generator,
            HEIGHT=self.HEIGHT,
            WIDTH=self.WIDTH,
            SIGMA=self.SIGMA,
            IMG_FORMAT=self.IMG_FORMAT,
            median=self.median,
            median_range=self.median_max_sample_num,
        )
        data_loader_tn = DataLoader(
            iterable_tn, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        num_sample_tn, sample_count_tn = total_frames - tracknet_seq_len + 1, 0
        buffer_size_tn = tracknet_seq_len - 1
        sample_indices_tn = torch.arange(tracknet_seq_len)
        frame_indices_tn = torch.arange(tracknet_seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros(
            (buffer_size_tn, tracknet_seq_len, self.HEIGHT, self.WIDTH), dtype=torch.float32
        )
        weight_tn = get_ensemble_weight(tracknet_seq_len, self.EVAL_MODE)

        tracknet_bbox_confidence_results: Dict[int, Dict] = {}
        tracknet_coord_results: Dict[int, Dict] = {}
        current_frame_index_tn = 0

        print("tracknet inference...")
        for x in tqdm(data_loader_tn):
            x = x.float().to(self.DEVICE)
            batch_size = x.shape[0]

            with torch.no_grad():
                tracknet_raw_y_pred = self.tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, tracknet_raw_y_pred), dim=0)
            batch_ensembled_raw_heatmaps = torch.empty(
                 (0, 1, self.HEIGHT, self.WIDTH), dtype=torch.float32
            )
            num_frames_output_in_batch = 0
            for sample_i in range(batch_size):
                 if sample_count_tn < buffer_size_tn:
                     ensembled_heatmap_raw = y_pred_buffer[
                         sample_indices_tn + sample_i, frame_indices_tn
                     ].sum(0) / (sample_count_tn + 1)
                 else:
                     ensembled_heatmap_raw = (
                         y_pred_buffer[sample_indices_tn + sample_i, frame_indices_tn]
                         * weight_tn[:, None, None]
                     ).sum(0)

                 batch_ensembled_raw_heatmaps = torch.cat(
                     (batch_ensembled_raw_heatmaps, ensembled_heatmap_raw.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0
                 )
                 num_frames_output_in_batch += 1
                 sample_count_tn += 1

                 if sample_count_tn == num_sample_tn:
                     y_zero_pad = torch.zeros(
                         (buffer_size_tn, tracknet_seq_len, self.HEIGHT, self.WIDTH), dtype=torch.float32
                     )
                     y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                     for frame_i in range(1, tracknet_seq_len):
                         ensembled_heatmap_raw_final = y_pred_buffer[
                             sample_indices_tn + sample_i + frame_i, frame_indices_tn
                         ].sum(0) / (tracknet_seq_len - frame_i)
                         batch_ensembled_raw_heatmaps = torch.cat(
                             (batch_ensembled_raw_heatmaps, ensembled_heatmap_raw_final.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0
                         )
                         num_frames_output_in_batch += 1

            batch_predictions_tn = predict_bbox_and_confidence(
                raw_y_pred=batch_ensembled_raw_heatmaps,
                img_scaler=img_scaler,
                threshold=heatmap_threshold
            )
            for i in range(num_frames_output_in_batch):
                frame_idx = current_frame_index_tn + i
                if frame_idx >= total_frames: break

                prediction_info = batch_predictions_tn[i]
                tracknet_bbox_confidence_results[frame_idx] = prediction_info

                if frame_idx == 59:
                    print(f"\n[DEBUG TrackNet Frame 59] BBox/Conf: {prediction_info}")

                vis = 0
                center_x, center_y = 0.0, 0.0
                if prediction_info['bbox'] is not None:
                    vis = 1
                    center_x = prediction_info['bbox'][0] + prediction_info['bbox'][2] / 2.0
                    center_y = prediction_info['bbox'][1] + prediction_info['bbox'][3] / 2.0

                tracknet_coord_results[frame_idx] = {'x': center_x, 'y': center_y, 'visibility': vis}


            current_frame_index_tn += num_frames_output_in_batch
            y_pred_buffer = y_pred_buffer[-buffer_size_tn:]

        final_coords_dict = {} 

        if self.inpaintnet is not None:
            print("Running InpaintNet inference and ensembling...")
            inpaintnet_input_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

            for i in range(total_frames):
                if i in tracknet_coord_results:
                    inpaintnet_input_dict['Frame'].append(i)
                    inpaintnet_input_dict['X'].append(tracknet_coord_results[i]['x'])
                    inpaintnet_input_dict['Y'].append(tracknet_coord_results[i]['y'])
                    inpaintnet_input_dict['Visibility'].append(tracknet_coord_results[i]['visibility'])
                else:
                    inpaintnet_input_dict['Frame'].append(i)
                    inpaintnet_input_dict['X'].append(0.0)
                    inpaintnet_input_dict['Y'].append(0.0)
                    inpaintnet_input_dict['Visibility'].append(0)
            inpaintnet_input_dict["Inpaint_Mask"] = generate_inpaint_mask(
                 {'X': inpaintnet_input_dict['X'], 'Y': inpaintnet_input_dict['Y'], 'Visibility': inpaintnet_input_dict['Visibility']}, # Pass dict with expected keys
                 th_h=self.video_info.height * 0.05
            )

            inpaintnet_input_dict['Img_scaler'] = img_scaler
            inpaintnet_input_dict['Img_shape'] = (self.video_info.width, self.video_info.height)


            inpaint_seq_len = self.inpaintnet_seq_len
            dataset_in = BallTrajectoryDataset(
                seq_len=inpaint_seq_len,
                sliding_step=1,
                data_mode='coordinate',
                pred_dict=inpaintnet_input_dict,
                HEIGHT=self.HEIGHT, WIDTH=self.WIDTH, SIGMA=self.SIGMA, IMG_FORMAT=self.IMG_FORMAT
            )
            data_loader_in = DataLoader(
                 dataset_in, batch_size=self.batch_size, shuffle=False, drop_last=False
            )

            self.inpaintnet.eval()
            self.inpaintnet.to(self.DEVICE)
            weight_in = get_ensemble_weight(inpaint_seq_len, self.EVAL_MODE)
            num_sample_in, sample_count_in = len(dataset_in), 0
            buffer_size_in = inpaint_seq_len - 1
            sample_indices_in = torch.arange(inpaint_seq_len)
            frame_indices_in = torch.arange(inpaint_seq_len - 1, -1, -1)
            coor_inpaint_buffer = torch.zeros(
                (buffer_size_in, inpaint_seq_len, 2), dtype=torch.float32
            )


            for (i_in, coor_pred_in, _coor_gt, _vis_pred, _vis_gt, inpaint_mask_in) in tqdm(data_loader_in):
                coor_pred_in, inpaint_mask_in = coor_pred_in.float().to(self.DEVICE), inpaint_mask_in.float().to(self.DEVICE)
                batch_size_in = i_in.shape[0]
                with torch.no_grad():
                    coor_inpaint_norm = self.inpaintnet(coor_pred_in, inpaint_mask_in).detach().cpu()
                coor_orig_norm = coor_pred_in.cpu()
                coor_refined_norm = coor_inpaint_norm * inpaint_mask_in.cpu() + coor_orig_norm * (1 - inpaint_mask_in.cpu())
                th_mask_norm = (coor_refined_norm[:, :, 0] < self.COOR_TH) & (coor_refined_norm[:, :, 1] < self.COOR_TH)
                coor_refined_norm[th_mask_norm] = 0.
                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_refined_norm), dim=0)
                ensemble_i_in = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint_norm = torch.empty((0, 1, 2), dtype=torch.float32)
                num_frames_output_in_batch_in = 0
                for sample_i in range(batch_size_in):
                     if sample_count_in < buffer_size_in:
                         coor_ensembled_norm = coor_inpaint_buffer[
                             sample_indices_in + sample_i, frame_indices_in
                         ].sum(0) / (sample_count_in + 1)
                     else:
                         coor_ensembled_norm = (
                             coor_inpaint_buffer[sample_indices_in + sample_i, frame_indices_in]
                             * weight_in[:, None]
                         ).sum(0)
                     original_frame_index = i_in[sample_i, 0, 1].item()
                     ensemble_i_in = torch.cat((ensemble_i_in, torch.tensor([[[0, original_frame_index]]])), dim=0)
                     ensemble_coor_inpaint_norm = torch.cat(
                         (ensemble_coor_inpaint_norm, coor_ensembled_norm.view(1, 1, 2)), dim=0
                     )
                     num_frames_output_in_batch_in += 1
                     sample_count_in += 1
                     if sample_count_in == num_sample_in:
                         coor_zero_pad = torch.zeros((buffer_size_in, inpaint_seq_len, 2), dtype=torch.float32)
                         coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                         for frame_i in range(1, inpaint_seq_len):
                              coor_ensembled_norm_final = coor_inpaint_buffer[
                                  sample_indices_in + sample_i + frame_i, frame_indices_in
                              ].sum(0) / (inpaint_seq_len - frame_i)
                              original_frame_index_final = i_in[sample_i, frame_i, 1].item()
                              ensemble_i_in = torch.cat((ensemble_i_in, torch.tensor([[[0, original_frame_index_final]]])), dim=0)
                              ensemble_coor_inpaint_norm = torch.cat(
                                  (ensemble_coor_inpaint_norm, coor_ensembled_norm_final.view(1, 1, 2)), dim=0
                              )
                              num_frames_output_in_batch_in += 1
                th_mask_final_norm = (ensemble_coor_inpaint_norm[:, :, 0] < self.COOR_TH) & (ensemble_coor_inpaint_norm[:, :, 1] < self.COOR_TH)
                ensemble_coor_inpaint_norm[th_mask_final_norm] = 0.
                ensemble_coor_inpaint_pixels = ensemble_coor_inpaint_norm.clone()
                ensemble_coor_inpaint_pixels[:, :, 0] *= self.video_info.width
                ensemble_coor_inpaint_pixels[:, :, 1] *= self.video_info.height
                for k in range(num_frames_output_in_batch_in):
                    frame_idx = int(ensemble_i_in[k, 0, 1].item())
                    if frame_idx >= total_frames: continue
                    if frame_idx not in final_coords_dict:
                        coords_pix = ensemble_coor_inpaint_pixels[k, 0, :].tolist()
                        cx_pred = int(coords_pix[0])
                        cy_pred = int(coords_pix[1])
                        vis_pred = 0 if (cx_pred == 0 and cy_pred == 0) else 1
                        final_coords_dict[frame_idx] = {'x': float(cx_pred), 'y': float(cy_pred), 'visibility': vis_pred}
                        if frame_idx == 59:
                            print(f"[DEBUG InpaintNet Frame 59] Coords/Vis: {final_coords_dict[frame_idx]}")
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size_in:]
        else:
            print("InpaintNet not used. Using TrackNet coordinates.")
            final_coords_dict = tracknet_coord_results

        evaluation_results = []
        print("Aggregating final results for evaluation...")
        for frame_idx in range(total_frames):
            final_coord_info = final_coords_dict.get(frame_idx, {'x': 0.0, 'y': 0.0, 'visibility': 0})
            final_x = final_coord_info['x']
            final_y = final_coord_info['y']
            final_visibility = final_coord_info['visibility']

            tracknet_info = tracknet_bbox_confidence_results.get(frame_idx, {'bbox': None, 'confidence': 0.0})
            bbox_tn = tracknet_info['bbox']
            conf_tn = tracknet_info['confidence']

            pred_bbox_eval = None
            pred_conf_eval = 0.0

            if final_visibility == 1 and bbox_tn is not None:
                 w_tn, h_tn = bbox_tn[2], bbox_tn[3]
                 x_tl_eval = final_x - w_tn / 2.0
                 y_tl_eval = final_y - h_tn / 2.0
                 pred_bbox_eval = [x_tl_eval, y_tl_eval, w_tn, h_tn]
                 pred_conf_eval = conf_tn

            if frame_idx == 59:
                print(f"final coords: {final_coord_info}")
                print(f"[TrackNet Info: {tracknet_info}")
                print(f"Eval BBox: {pred_bbox_eval}")
                print(f"Eval Conf: {pred_conf_eval}")

            evaluation_results.append({
                'frame_index': frame_idx,
                'pred_bbox': pred_bbox_eval,
                'pred_conf': pred_conf_eval
            })

        print("eval complete")
        return evaluation_results