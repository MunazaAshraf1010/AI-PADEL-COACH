input_video = "./input_video.mp4"

output_video = "results.mp4"

save_data = True
save_data_path = "data_file.csv"
report_output_path = "match_report.json"
highlights_output_dir = "highlights/"

player_tracking_model = "./weights/players_detection/yolov8m.pt"
player_tracking_batch_sz = 20
player_tracking_annt_type = "rectangle_bounding_box"
# Prediction cache: load+save point at the same file so each tracker is computed
# once and reused on later runs. Delete ./cache/ if you change input_video or
# maximum_frame_count (predictions are indexed by frame and would otherwise be stale).
player_tracking_load = "./cache/players.json"
player_tracking_save = "./cache/players.json"

maximum_frame_count = None  # None = full video (33,422 frames / 22 min), or set a number for quick test

court_keypoint_path =  None
court_keypoint_save_path = None 

player_keypoint_model = "./weights/players_keypoints_detection/model.pt"
player_keypoint_img_sz = 1280
player_keypoint_batch_sz = 20
player_keypoint_tracking_path = "./cache/player_keypoints.json"
player_keypoint_tracing_save = "./cache/player_keypoints.json"

ball_tracker_model = "./weights/ball_detection/TrackNet_best.pt"
inpaint_model = "./weights/ball_detection/InpaintNet_best.pt"
ball_tracking_batch_sz = 20
ball_tracking_sample_num = 400
ball_tracking_path = "./cache/ball.json"
ball_tracking_save = "./cache/ball.json"

keypoint_tracking_model = "./weights/court_keypoints_detection/model.pt"
keypoint_tracking_batch_sz = 20
keypoint_tracking_model_type = "yolo"
keypoint_tracking_path = "./cache/court_keypoints.json"
keypoint_tracking_save = "./cache/court_keypoints.json"

# Dashboard settings
launch_dashboard = True
dashboard_port = 8086

