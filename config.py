input_video = "./input_video.mp4"

output_video = "results.mp4"

save_data = True
save_data_path = "data_file.csv"
report_output_path = "match_report.json"
highlights_output_dir = "highlights/"

player_tracking_model = "./weights/players_detection/yolov8m.pt"
player_tracking_batch_sz = 20
player_tracking_annt_type = "rectangle_bounding_box"
player_tracking_load = None
player_tracking_save = None 

maximum_frame_count = None  # None = full video (33,422 frames / 22 min), or set a number for quick test

court_keypoint_path =  None
court_keypoint_save_path = None 

player_keypoint_model = "./weights/players_keypoints_detection/model.pt"
player_keypoint_img_sz = 1280
player_keypoint_batch_sz = 20
player_keypoint_tracking_path = None 
player_keypoint_tracing_save = None

ball_tracker_model = "./weights/ball_detection/TrackNet_best.pt"
inpaint_model = "./weights/ball_detection/InpaintNet_best.pt"
ball_tracking_batch_sz = 20
ball_tracking_sample_num = 400
ball_tracking_path = None 
ball_tracking_save = None

keypoint_tracking_model = "./weights/court_keypoints_detection/model.pt"
keypoint_tracking_batch_sz = 20
keypoint_tracking_model_type = "yolo"
keypoint_tracking_path = None 
keypoint_tracking_save = None

# Dashboard settings
launch_dashboard = True
dashboard_port = 8080

