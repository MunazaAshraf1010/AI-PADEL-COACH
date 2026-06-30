"""
Regenerate match_report.json from cached tracker predictions.

The full pipeline (main.py) re-decodes the 1GB video and rewrites a 4GB
annotated video just to feed the analytics. All detection/tracking results are
already cached under ./cache, and the court→canvas projection is pure geometry
(homography), so we can rebuild the report by replaying the cache straight into
ComprehensiveStats — no model weights, no GPU, no video decoding.

This mirrors Runner._feed_comprehensive_stats exactly so the numbers match what
a real run would now produce with the fixed analytics.
"""

import json
import sys
import supervision as sv

# Import the trackers package first: analytics.court_projection imports from
# trackers, which imports the Runner, which imports analytics — importing
# analytics first trips that circular chain. main.py imports trackers first too.
from trackers import Players, Ball, Keypoints
from trackers.players_keypoints_tracker.players_keypoints_tracker import PlayersKeypoints

from config import (
    input_video,
    report_output_path,
    player_tracking_load,
    ball_tracking_path,
    keypoint_tracking_path,
    player_keypoint_tracking_path,
)
from constants import COURT_LENGTH, COURT_WIDTH
from analytics.court_projection import ProjectedCourt
from analytics.comprehensive_stats import ComprehensiveStats


def _load(path, object_cls):
    with open(path, "r") as f:
        raw = json.load(f)
    return [object_cls.from_json(entry) for entry in raw]


def main():
    print("regenerate: reading video metadata ...")
    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    fps = float(video_info.fps)
    print(f"regenerate: {video_info.width}x{video_info.height} @ {fps:.3f} fps")

    projected_court = ProjectedCourt(video_info)
    cp = projected_court.court_position

    def canvas_to_court_meters(projection):
        x_m = (float(projection[0]) - cp.top_left[0]) / cp.width * COURT_WIDTH
        y_m = (float(projection[1]) - cp.top_left[1]) / cp.height * COURT_LENGTH
        return (x_m, y_m)

    print("regenerate: loading cached predictions ...")
    players_all = _load(player_tracking_load, Players)
    ball_all = _load(ball_tracking_path, Ball)
    keypoints_all = _load(keypoint_tracking_path, Keypoints)
    try:
        pk_all = _load(player_keypoint_tracking_path, PlayersKeypoints)
    except (FileNotFoundError, OSError):
        pk_all = []

    n = min(len(players_all), len(ball_all), len(keypoints_all))
    print(f"regenerate: players={len(players_all)} ball={len(ball_all)} "
          f"court_kp={len(keypoints_all)} player_kp={len(pk_all)} -> replaying {n} frames")

    # Court keypoints are fixed for this video; compute the homography once,
    # matching the pipeline's "first homography" behaviour.
    H = None
    for kp in keypoints_all:
        if kp and len(kp) in (12, 18, 22):
            H = projected_court.homography_matrix(kp)
            break
    if H is None:
        print("regenerate: ERROR could not compute homography from court keypoints")
        sys.exit(1)

    stats = ComprehensiveStats(
        court_length=COURT_LENGTH,
        court_width=COURT_WIDTH,
        fps=fps,
    )

    for frame_index in range(n):
        players_detection = players_all[frame_index]
        ball_detection = ball_all[frame_index]
        pk_detection = pk_all[frame_index] if frame_index < len(pk_all) else None

        # --- project + collect player positions (court metres) ---
        players_positions = {}
        for player in players_detection:
            if player.id is None or player.id > 4:
                continue
            projected_court.project_player(player, H)
            if player.projection is not None:
                players_positions[player.id] = canvas_to_court_meters(player.projection)

        # --- player pose keypoints (index-aligned, same as Runner) ---
        players_keypoints = {}
        if pk_detection is not None:
            for i, player in enumerate(players_detection):
                if player.id is not None and player.id <= 4 and i < len(pk_detection):
                    try:
                        kp_dict = {kp.name: kp.xy for kp in pk_detection[i].player_keypoints}
                        players_keypoints[player.id] = kp_dict
                    except (IndexError, AttributeError):
                        pass

        # --- ball position (court metres), gated on visibility ---
        ball_position = None
        projected_court.project_ball(ball_detection, H)
        if ball_detection.projection is not None and getattr(ball_detection, "visibility", 1):
            ball_position = canvas_to_court_meters(ball_detection.projection)

        stats.process_frame(
            frame_index=frame_index,
            players_positions=players_positions or None,
            players_keypoints=players_keypoints or None,
            ball_position=ball_position,
        )

        if frame_index % 5000 == 0:
            print(f"  ... frame {frame_index}/{n}")

    print("regenerate: writing report ...")
    stats.save_report(report_output_path, input_video)
    stats.print_summary()


if __name__ == "__main__":
    main()
