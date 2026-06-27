from trackers import (
    TRPlayer, 
    TRBall, 
    TrKeypoints, 
    Keypoint,
    Keypoints,
    TRPlayerKeypoint,
    Runner,
)
from config import *
import timeit
import json
import cv2
import numpy as np
import supervision as sv

selected_kp = []

if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.0f}s)")

    # --- Court keypoints detection ---
    if court_keypoint_path is not None:
        with open(court_keypoint_path, "r") as f:
            selected_kp = json.load(f)
    else:
        # Auto-detect court keypoints from first gameplay frame
        print("Auto-detecting court keypoints from video...")
        from ultralytics import YOLO
        kp_model = YOLO(keypoint_tracking_model)
        
        # Use frame 100 (skip intro screens)
        frame_gen = sv.get_video_frames_generator(input_video, start=100, end=101)
        first_frame = next(frame_gen)
        from PIL import Image
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(first_frame_rgb).resize((640, 640))
        
        kp_results = kp_model.predict(img, conf=0.5, iou=0.7, imgsz=640, max_det=12)
        
        h_frame, w_frame = first_frame.shape[:2]
        ratio_x = w_frame / 640
        ratio_y = h_frame / 640
        
        points_mapper = {0: 10, 1: 11, 2: 1, 3: 0, 4: 7, 5: 9, 6: 8, 7: 5, 8: 6, 9: 2, 10: 4, 11: 3}
        
        detected_kp = []
        if len(kp_results) > 0 and kp_results[0].keypoints is not None:
            kp_xy = kp_results[0].keypoints.xy
            if kp_xy.ndim == 3:
                kp_xy = kp_xy.squeeze(0)
            for i, kp_det in enumerate(kp_xy):
                mapped_id = points_mapper.get(i, i)
                detected_kp.append((mapped_id, (float(kp_det[0]) * ratio_x, float(kp_det[1]) * ratio_y)))
            detected_kp.sort(key=lambda x: x[0])
            selected_kp = [list(kp[1]) for kp in detected_kp]
            print(f"  Auto-detected {len(selected_kp)} court keypoints")
        
        del kp_model

    # Fallback if auto-detection found too few keypoints
    if len(selected_kp) < 4:
        print("  Using default court keypoints for standard padel court view...")
        selected_kp = [
            [int(w * 0.05), int(h * 0.95)],   # k1: bottom-left
            [int(w * 0.95), int(h * 0.95)],   # k2: bottom-right
            [int(w * 0.10), int(h * 0.60)],   # k3
            [int(w * 0.50), int(h * 0.60)],   # k4
            [int(w * 0.90), int(h * 0.60)],   # k5
            [int(w * 0.15), int(h * 0.40)],   # k6
            [int(w * 0.85), int(h * 0.40)],   # k7
            [int(w * 0.20), int(h * 0.25)],   # k8
            [int(w * 0.50), int(h * 0.25)],   # k9
            [int(w * 0.80), int(h * 0.25)],   # k10
            [int(w * 0.25), int(h * 0.10)],   # k11: top-left
            [int(w * 0.75), int(h * 0.10)],   # k12: top-right
        ]

    if court_keypoint_save_path is not None:
        with open(court_keypoint_save_path, "w") as f:
            json.dump(selected_kp, f)

    key_point_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(selected_kp)
        ]
    )

    keypoints_array = np.array(selected_kp)
    print(f"Court keypoints: {keypoints_array.shape}")
    
    if len(keypoints_array) < 4:
        raise ValueError("Not enough keypoints to define court polygon")

    polygon_zone = sv.PolygonZone(
        np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0), 
                np.expand_dims(keypoints_array[1], axis=0), 
                np.expand_dims(keypoints_array[-1], axis=0), 
                np.expand_dims(keypoints_array[-2], axis=0),
            ),
            axis=0
        ),
    )

    # --- Initialize all trackers ---
    players_tracker = TRPlayer(
        player_tracking_model,
        polygon_zone,
        batch_size=player_tracking_batch_sz,
        annotator=player_tracking_annt_type,
        show_confidence=True,
        load_path=player_tracking_load,
        save_path=player_tracking_save,
    )

    player_keypoints_tracker = TRPlayerKeypoint(
        player_keypoint_model,
        train_image_size=player_keypoint_img_sz,
        batch_size=player_keypoint_batch_sz,
        load_path=player_keypoint_tracking_path,
        save_path=player_keypoint_tracing_save,
    )

    ball_tracker = TRBall(
        ball_tracker_model,
        inpaint_model,
        batch_size=ball_tracking_batch_sz,
        median_max_sample_num=ball_tracking_sample_num,
        median=None,
        load_path=ball_tracking_path,
        save_path=ball_tracking_save,
    )

    keypoints_tracker = TrKeypoints(
        model_path=keypoint_tracking_model,
        batch_size=keypoint_tracking_batch_sz,
        model_type=keypoint_tracking_model_type,
        fixed_keypoints_detection=key_point_detection,
        load_path=keypoint_tracking_path,
        save_path=keypoint_tracking_save,
    )

    # --- Run pipeline ---
    runner = Runner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,    
        ],
        video_path=input_video,
        inference_path=output_video,
        start=0,
        end=maximum_frame_count,
        collect_data=save_data,
    )

    runner.run()

    if save_data:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(save_data_path)

    # Generate comprehensive analytics report
    runner.generate_comprehensive_report(output_path=report_output_path)

    t2 = timeit.default_timer()
    print("duration: ", (t2 - t1) / 60)

    # Launch web dashboard
    if launch_dashboard:
        from dashboard_server import start_dashboard
        start_dashboard(
            report_path=report_output_path,
            port=dashboard_port,
            open_browser=True,
        )
