from typing import List, Union, Optional, Dict
import numpy as np
import cv2
import torch


def predict_location(heatmap: np.array):
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    else:
        (counts, _) = cv2.findContours(
            heatmap.copy(), 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        rects = [cv2.boundingRect(ctr) for ctr in counts]
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area

        x, y, w, h = rects[max_area_idx]

        return x, y, w, h
    

def to_img(image):
    image = image * 255
    image = image.astype('uint8')
    return image


def to_img_format(input, WIDTH: int, HEIGHT: int, num_ch=1):
    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    
    if num_ch == 1:
        return input
    else:
        input = np.transpose(input, (0, 2, 3, 1))
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3)
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq



def predict(indices, WIDTH: int, HEIGHT: int, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred, WIDTH=WIDTH, HEIGHT=HEIGHT) # (N, L, H, W)
    
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    


def predict_modified(
    WIDTH: int, 
    HEIGHT: int, 
    y_pred: Union[torch.Tensor, np.array] = None, 
    c_pred: Union[torch.Tensor, np.array] = None, 
    img_scaler: tuple[float, float] = (1.0, 1.0),
    threshold: float = 0.5,
) -> dict:
    pred_dict = {
        'x': [], 
        'y': [], 
        'visibility': []
    }
    
    if y_pred is not None:
        y_pred = y_pred > threshold
        y_pred = (
            y_pred.detach().cpu().numpy() 
            if torch.is_tensor(y_pred) 
            else y_pred
        )
    if c_pred is not None:
        c_pred = (
            c_pred.detach().cpu().numpy() 
            if torch.is_tensor(c_pred) 
            else c_pred
        )

    number_preds = y_pred.shape[0]
    for n in range(number_preds):
        if c_pred is not None:
            c_p = c_pred[n][0]
            cx_pred, cy_pred = (
                int(c_p[0] * WIDTH * img_scaler[0]), 
                int(c_p[1] * HEIGHT * img_scaler[1]),
            )
        elif y_pred is not None:
            y_p = y_pred[n][0]
            bbox_pred = predict_location(to_img(y_p))
            cx_pred, cy_pred = (
                int(bbox_pred[0]+bbox_pred[2]/2), 
                int(bbox_pred[1]+bbox_pred[3]/2),
            )
            cx_pred, cy_pred = (
                int(cx_pred*img_scaler[0]), 
                int(cy_pred*img_scaler[1]),
            )
        else:
            raise ValueError('Invalid input')
        
        viz_pred = 0 if (cx_pred == 0 and cy_pred == 0) else 1
        pred_dict["x"].append(cx_pred)
        pred_dict["y"].append(cy_pred)
        pred_dict["visibility"].append(viz_pred)
    
    return pred_dict

def predict_bbox_and_confidence(
    raw_y_pred: Union[torch.Tensor, np.array],
    img_scaler: tuple[float, float] = (1.0, 1.0),
    threshold: float = 0.3,
) -> List[Dict[str, Optional[Union[List[float], float]]]]:
    results = []
    batch_size = raw_y_pred.shape[0]

    raw_y_pred_np = (
        raw_y_pred.detach().cpu().numpy()
        if torch.is_tensor(raw_y_pred)
        else raw_y_pred
    )

    for n in range(batch_size):

        if raw_y_pred_np.ndim == 4:
             heatmap_raw = raw_y_pred_np[n][0]
        elif raw_y_pred_np.ndim == 3:
             heatmap_raw = raw_y_pred_np[n]
        else:
             raise ValueError(f"Unsupported raw_y_pred shape: {raw_y_pred_np.shape}")

        confidence = np.max(heatmap_raw)
        final_confidence = float(confidence)
        heatmap_thresh = heatmap_raw > threshold

        if heatmap_thresh.dtype == bool:
            heatmap_thresh_img = heatmap_thresh.astype(np.uint8) * 255
        else:
            heatmap_thresh_img = (heatmap_thresh * 255).astype(np.uint8)
        bbox_model_dims = predict_location(heatmap_thresh_img)
        pred_bbox_scaled = None

        if bbox_model_dims != (0, 0, 0, 0):
            x, y, w, h = bbox_model_dims
            x_s = x * img_scaler[0]
            y_s = y * img_scaler[1]
            w_s = w * img_scaler[0]
            h_s = h * img_scaler[1]
            pred_bbox_scaled = [float(x_s), float(y_s), float(w_s), float(h_s)]
        results.append({'bbox': pred_bbox_scaled, 'confidence': final_confidence})

    return results