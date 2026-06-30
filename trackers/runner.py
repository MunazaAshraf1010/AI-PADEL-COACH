""" 
Implementation of a runner to extract results from an arbitrary list of trackers 
"""

from typing import Optional
from tqdm import tqdm
import timeit
from copy import deepcopy
from pathlib import Path
import cv2
import supervision as sv

from trackers.players_tracker.players_tracker import Players
from trackers.ball_tracker.ball_tracker import Ball
from trackers.keypoints_tracker.keypoints_tracker import Keypoints
from trackers.players_keypoints_tracker.players_keypoints_tracker import PlayersKeypoints
from trackers.tracker import Tracker
from analytics.court_projection import ProjectedCourt
from analytics.data_analysis import DataAnalytics


class Runner:
    def __init__(
        self, 
        trackers: list[Tracker],
        video_path: str | Path,
        inference_path: str | Path,
        start: int = 0,
        end: Optional[int] = None,
        collect_data: bool = False, 
    ) -> None:
    
        self.video_path = video_path
        self.inference_path = inference_path
        self.start = start
        self.stride = 1
        self.end = end
        self.video_info = sv.VideoInfo.from_video_path(video_path=video_path)

        if self.end is None:
            self.total_frames = self.video_info.total_frames
        else:
            self.total_frames = self.end - self.start

        self.trackers = {}
        self.is_fixed_keypoints = False
        for tracker in trackers:
            self.trackers[str(tracker)] = tracker.video_info_post_init(self.video_info)

            if tracker.object() == Keypoints:
                self.is_fixed_keypoints = not(
                    tracker.fixed_keypoints_detection is None
                )
        
        if self.is_fixed_keypoints:
            print("-"*40)
            print("runner: using fixed court keypoints")
            print("-"*40)

        self.projected_court = ProjectedCourt(self.video_info)
        if collect_data:
            print("runner: ready for data collection")
            self.data_analytics = DataAnalytics()
        else:
            self.data_analytics = None

        # Initialize comprehensive analytics
        from analytics.comprehensive_stats import ComprehensiveStats
        from constants import COURT_LENGTH, COURT_WIDTH
        self.comprehensive_stats = ComprehensiveStats(
            court_length=COURT_LENGTH,
            court_width=COURT_WIDTH,
            fps=float(self.video_info.fps),
        )
    
    def restart(self) -> None:
        for tracker in self.trackers.values():
            tracker.restart()
        
        if self.data_analytics:
            self.data_analytics.restart()

    def draw_and_collect_data(self) -> None:

        print(f"runner: writing results into {str(self.inference_path)}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.inference_path,
            fourcc,
            float(self.video_info.fps),
            self.video_info.resolution_wh,
        )

        frame_generator = sv.get_video_frames_generator(
            self.video_path,
            start=self.start,
            stride=self.stride,
            end=self.end,
        )

        for frame_index, frame in tqdm(enumerate(frame_generator)):
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(
                frame_rgb,
                f"Frame: {frame_index + 1}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                1,
            )

            players_detection = None
            ball_detection = None
            keypoints_detection = None
            players_keypoints_detection = None
            for tracker in self.trackers.values():
                
                try:
                    prediction = tracker.results[frame_index]
                except IndexError as e:
                    print(f"runner: {str(tracker)} frame {frame_index}")
                    raise(e)
                
                frame_rgb = prediction.draw(frame_rgb, **tracker.draw_kwargs())

                if tracker.object() == Players:
                    players_detection = deepcopy(prediction)
                elif tracker.object() == Ball:
                    ball_detection = deepcopy(prediction)
                elif tracker.object() == Keypoints:
                    keypoints_detection = deepcopy(prediction)
                elif tracker.object() == PlayersKeypoints:
                    players_keypoints_detection = deepcopy(prediction)
               
            output_frame, self.data_analytics = self.projected_court.draw_projections_and_collect_data(
                frame_rgb,
                keypoints_detection=keypoints_detection,
                players_detection=players_detection,
                ball_detection=ball_detection,
                data_analytics=self.data_analytics,
                is_fixed_keypoints=self.is_fixed_keypoints,
            )

            # Feed data into comprehensive analytics
            self._feed_comprehensive_stats(
                frame_index=frame_index,
                players_detection=players_detection,
                ball_detection=ball_detection,
                players_keypoints_detection=players_keypoints_detection,
            )

            if self.data_analytics is not None:
                self.data_analytics.step(1)

            out.write(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        
        out.release()
        self.data_analytics.frames = self.data_analytics.frames[:-1]

        print("runner: Done") 

    def _canvas_to_court_meters(self, projection) -> tuple[float, float]:
        """Convert a mini-court canvas-pixel projection to 0-based court metres.

        The projected court occupies `court_position` on the canvas: its pixel
        width maps to COURT_WIDTH (10m) and its pixel height to COURT_LENGTH (20m).
        """
        from constants import COURT_LENGTH, COURT_WIDTH
        cp = self.projected_court.court_position
        x_m = (float(projection[0]) - cp.top_left[0]) / cp.width * COURT_WIDTH
        y_m = (float(projection[1]) - cp.top_left[1]) / cp.height * COURT_LENGTH
        return (x_m, y_m)

    def _feed_comprehensive_stats(
        self,
        frame_index: int,
        players_detection,
        ball_detection,
        players_keypoints_detection=None,
    ):
        """
        Extract projected positions from detections and feed into 
        ComprehensiveStats for advanced analytics.
        """
        from constants import COURT_LENGTH, COURT_WIDTH
        
        # Frame dimensions for coordinate normalization
        frame_w = self.video_info.resolution_wh[0]
        frame_h = self.video_info.resolution_wh[1]
        
        # Extract player positions - normalize pixel coords to court meters
        players_positions = {}
        if players_detection is not None and len(players_detection) > 0:
            for player in players_detection:
                if player.id is None:
                    continue
                # Only track up to 4 players (standard padel)
                if player.id > 4:
                    continue
                    
                if player.projection is not None:
                    # `projection` is in mini-court CANVAS pixels; convert to
                    # 0-based court metres so it matches the units the analytics
                    # modules expect (x in [0, COURT_WIDTH], y in [0, COURT_LENGTH]).
                    players_positions[player.id] = self._canvas_to_court_meters(
                        player.projection
                    )
                # No homography projection this frame -> skip rather than feed
                # bogus pixel-normalised coordinates into metre-based analytics.

        # Extract player keypoints
        players_keypoints = {}
        if players_keypoints_detection is not None and players_detection is not None:
            for i, player in enumerate(players_detection):
                if player.id is not None and player.id <= 4 and i < len(players_keypoints_detection):
                    try:
                        player_kp = players_keypoints_detection[i]
                        kp_dict = {}
                        for kp in player_kp.player_keypoints:
                            kp_dict[kp.name] = kp.xy
                        players_keypoints[player.id] = kp_dict
                    except (IndexError, AttributeError):
                        pass

        # Extract ball position (projected court metres). Only feed it when the
        # ball is actually visible and projected - an invisible ball has xy=(0,0)
        # which would otherwise project to a bogus court location and trigger
        # phantom "hits".
        ball_position = None
        if (
            ball_detection is not None
            and getattr(ball_detection, 'projection', None) is not None
            and getattr(ball_detection, 'visibility', 1)
        ):
            ball_position = self._canvas_to_court_meters(ball_detection.projection)

        # Feed into comprehensive stats
        self.comprehensive_stats.process_frame(
            frame_index=frame_index,
            players_positions=players_positions if players_positions else None,
            players_keypoints=players_keypoints if players_keypoints else None,
            ball_position=ball_position,
        )

    def generate_comprehensive_report(self, output_path: str = "match_report.json"):
        """Generate and save the comprehensive analytics report"""
        print("runner: Generating comprehensive analytics report...")
        report = self.comprehensive_stats.save_report(output_path, self.video_path)
        self.comprehensive_stats.print_summary()
        return report


    def run(self) -> None:

        print(f"runner: Running {self.total_frames} frames")

        for tracker in self.trackers.values():

            if len(tracker) != 0:
                print(f"{tracker.__str__()}: {len(tracker)} predictions stored")
                

                continue

            tracker.to(tracker.DEVICE)
            print(f"{str(tracker)}: Running on {tracker.DEVICE} ...")

            frame_generator = sv.get_video_frames_generator(
                self.video_path,
                start=self.start,
                stride=self.stride,
                end=self.end,
            )

            t0 = timeit.default_timer()

            tracker.predict_and_update(
                frame_generator, 
                total_frames=self.total_frames,
            )
            t1 = timeit.default_timer()

            tracker.to("cpu")

            print(f"{str(tracker)}: {t1 - t0} inference time.")

            tracker.save_predictions()
        
        self.draw_and_collect_data()

        

    

    


        



    
    


