"""
Comprehensive Stats Orchestrator
Main aggregator that coordinates all analytics modules and produces 
the final stats report from video detection and tracking data.
"""

from typing import Optional
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path

from analytics.shot_detection import ShotDetector, ShotType
from analytics.rally_analysis import RallyAnalyzer
from analytics.court_zones import CourtZoneAnalyzer
from analytics.movement_analysis import MovementAnalyzer
from analytics.pressure_stats import PressureAnalyzer
from analytics.player_rating import PlayerRating
from analytics.strategy_insights import StrategyAnalyzer
from analytics.highlights import HighlightGenerator


class ComprehensiveStats:
    """
    Orchestrates all analytics modules to produce complete match statistics.
    
    Connects to the existing tracking pipeline:
    - Player detections (positions from homography projection)
    - Player keypoints (pose estimation)
    - Ball tracking (position per frame)
    - Court keypoints (for coordinate transformation)
    
    Produces stats categories:
    1. Shot stats (forehand, backhand, smash, volley, lob)
    2. Shot speed (speed of shots / serve speed)
    3. Shot accuracy (accuracy rates, placement quality)
    4. Rally stats (rally counts, length, seconds per point, hits per point)
    5. Court coverage (time in volley/transition/back zones)
    6. Movement/positioning (patterns, coverage, speed)
    7. Team positioning (doubles horizontal and vertical coverage)
    8. Shot distribution (by type and court area)
    9. Pressure stats (% forcing >2m opponent movement)
    10. Serve/return stats (serve speed, quality, return effectiveness)
    11. Errors/hits (hits and errors per player)
    12. Player level (AI rating 0-7)
    13. Strategy insights (winning strategies, strengths, weaknesses)
    14. AI recommendations (personalized tips)
    15. Video highlights (automatic match summary and clip info)
    """

    def __init__(
        self,
        court_length: float = 20.0,
        court_width: float = 10.0,
        fps: float = 30.0,
        pixels_per_meter: float = 1.0,
    ):
        self.court_length = court_length
        self.court_width = court_width
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter

        # Initialize all analyzers
        self.shot_detector = ShotDetector(court_length, court_width)
        self.rally_analyzer = RallyAnalyzer(court_length, fps)
        self.zone_analyzer = CourtZoneAnalyzer(court_length, court_width)
        self.movement_analyzer = MovementAnalyzer(court_length, court_width, fps)
        self.pressure_analyzer = PressureAnalyzer(court_length, court_width)
        self.player_rating = PlayerRating()
        self.strategy_analyzer = StrategyAnalyzer()
        self.highlight_generator = HighlightGenerator(fps)

        # Frame-level tracking
        self._frame_count = 0
        self._ball_positions_history: list[tuple[int, float, float]] = []
        self._last_hitting_player: Optional[int] = None
        self._serve_detected = False
        self._all_player_positions: dict[int, tuple[float, float]] = {}

    def reset(self):
        """Reset all analyzers"""
        self.shot_detector.reset()
        self.rally_analyzer.reset()
        self.zone_analyzer.reset()
        self.movement_analyzer.reset()
        self.pressure_analyzer.reset()
        self.player_rating.reset()
        self.strategy_analyzer.reset()
        self.highlight_generator.reset()
        self._frame_count = 0
        self._ball_positions_history = []
        self._last_hitting_player = None
        self._serve_detected = False
        self._all_player_positions = {}

    def process_frame(
        self,
        frame_index: int,
        players_positions: Optional[dict[int, tuple[float, float]]] = None,
        players_keypoints: Optional[dict[int, dict[str, tuple[float, float]]]] = None,
        ball_position: Optional[tuple[float, float]] = None,
    ):
        """
        Process a single frame's detection/tracking results.
        
        Args:
            frame_index: current frame number
            players_positions: {player_id: (x, y)} projected court positions
            players_keypoints: {player_id: {"head": (x,y), "right_hand": (x,y), ...}}
            ball_position: (x, y) ball position in court coordinates
        """
        self._frame_count = frame_index

        # Update player positions
        if players_positions:
            self._all_player_positions = players_positions
            for player_id, position in players_positions.items():
                # Movement analysis
                self.movement_analyzer.update(frame_index, player_id, position)
                # Zone analysis
                self.zone_analyzer.update(frame_index, player_id, position)

        # Update ball tracking
        if ball_position:
            self._ball_positions_history.append(
                (frame_index, ball_position[0], ball_position[1])
            )
            # Keep history manageable
            if len(self._ball_positions_history) > 300:
                self._ball_positions_history = self._ball_positions_history[-200:]

        # Rally tracking
        self.rally_analyzer.update(
            frame=frame_index,
            ball_position=ball_position,
            hitting_player_id=self._detect_hitting_player(
                ball_position, players_positions
            ),
        )

        # Shot detection (when ball is near a player)
        hitting_player = self._detect_hitting_player(ball_position, players_positions)
        if hitting_player is not None and ball_position is not None and players_positions:
            player_pos = players_positions.get(hitting_player)
            player_kp = players_keypoints.get(hitting_player) if players_keypoints else None

            if player_pos:
                shot = self.shot_detector.detect_shot(
                    frame=frame_index,
                    player_id=hitting_player,
                    player_keypoints=player_kp,
                    player_position=player_pos,
                    ball_position=ball_position,
                    ball_positions_history=self._ball_positions_history,
                    fps=self.fps,
                    pixels_per_meter=self.pixels_per_meter,
                    is_serving=self._is_serve_situation(frame_index),
                )

                if shot:
                    # Pressure analysis
                    self.pressure_analyzer.update(
                        frame=frame_index,
                        shot_player_id=hitting_player,
                        ball_position=ball_position,
                        all_player_positions=players_positions,
                    )

                    # Highlight detection for fast shots/smashes
                    if shot.speed_kmh >= self.highlight_generator.HIGH_SPEED_THRESHOLD:
                        self.highlight_generator.add_fast_shot(
                            frame_index, hitting_player,
                            shot.shot_type.value, shot.speed_kmh
                        )
                    if shot.shot_type == ShotType.SMASH:
                        self.highlight_generator.add_smash(
                            frame_index, hitting_player, shot.speed_kmh
                        )

                    self._last_hitting_player = hitting_player

    def _detect_hitting_player(
        self,
        ball_position: Optional[tuple[float, float]],
        players_positions: Optional[dict[int, tuple[float, float]]],
    ) -> Optional[int]:
        """
        Detect which player is hitting the ball based on proximity.
        Returns player_id if ball is close enough to a player, None otherwise.
        """
        if ball_position is None or not players_positions:
            return None

        HIT_DISTANCE_THRESHOLD = 2.0  # meters

        closest_player = None
        min_distance = float('inf')

        for player_id, pos in players_positions.items():
            dx = ball_position[0] - pos[0]
            dy = ball_position[1] - pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_player = player_id

        if min_distance <= HIT_DISTANCE_THRESHOLD:
            # Only register if player changed (avoid repeated hits)
            if closest_player != self._last_hitting_player:
                return closest_player

        return None

    def _is_serve_situation(self, frame_index: int) -> bool:
        """Detect if current situation is likely a serve"""
        # Simple heuristic: first hit in a rally
        if self.rally_analyzer.current_rally is None:
            return True
        if self.rally_analyzer.current_rally.hits == 0:
            return True
        return False

    def generate_report(self, video_path: Optional[str] = None) -> dict:
        """
        Generate the complete analytics report.
        
        Returns comprehensive dict with all stat categories.
        """
        # Collect stats from all modules
        shot_stats = self.shot_detector.get_shot_stats()
        rally_stats = self.rally_analyzer.get_rally_stats()
        zone_stats = self.zone_analyzer.get_zone_stats(self.fps)
        movement_stats = self.movement_analyzer.get_movement_stats()
        pressure_stats = self.pressure_analyzer.get_pressure_stats()

        # Generate highlights from collected data
        self.highlight_generator.generate_from_stats(
            shots=self.shot_detector.shots,
            rallies=self.rally_analyzer.rallies,
        )

        # Player ratings
        ratings = self.player_rating.get_all_ratings(
            shot_stats=shot_stats,
            movement_stats=movement_stats,
            zone_stats=zone_stats,
            rally_stats=rally_stats,
            pressure_stats=pressure_stats,
        )

        # Strategy insights and recommendations
        strategy = self.strategy_analyzer.analyze(
            shot_stats=shot_stats,
            movement_stats=movement_stats,
            zone_stats=zone_stats,
            rally_stats=rally_stats,
            pressure_stats=pressure_stats,
            player_ratings=ratings,
        )

        # Build comprehensive report
        report = {
            "match_info": {
                "total_frames_analyzed": self._frame_count,
                "duration_seconds": round(self._frame_count / self.fps, 1) if self.fps > 0 else 0,
                "fps": self.fps,
                "court_dimensions": {
                    "length_m": self.court_length,
                    "width_m": self.court_width,
                },
            },
            "shot_stats": {
                "summary": {
                    "total_shots": shot_stats["total_shots"],
                    "shot_types": shot_stats["shot_distribution"],
                },
                "by_type": self._get_shots_by_type(shot_stats),
                "shot_distribution": shot_stats["shot_distribution"],
            },
            "shot_speed": {
                "avg_speed_kmh": shot_stats["avg_speed_kmh"],
                "max_speed_kmh": shot_stats["max_speed_kmh"],
                "per_player": {
                    pid: {
                        "avg_speed_kmh": pdata.get("avg_speed_kmh", 0),
                        "max_speed_kmh": pdata.get("max_speed_kmh", 0),
                    }
                    for pid, pdata in shot_stats.get("player_stats", {}).items()
                },
            },
            "shot_accuracy": {
                "overall_accuracy": shot_stats["avg_accuracy"],
                "per_player": {
                    pid: {"accuracy": pdata.get("avg_accuracy", 0)}
                    for pid, pdata in shot_stats.get("player_stats", {}).items()
                },
            },
            "rally_stats": rally_stats,
            "court_coverage": zone_stats,
            "movement_positioning": movement_stats,
            "team_positioning": {
                "team_1": movement_stats.get("team_stats", {}).get("team_1", {}),
                "team_2": movement_stats.get("team_stats", {}).get("team_2", {}),
            },
            "pressure_stats": pressure_stats,
            "serve_return_stats": self._extract_serve_return_stats(shot_stats),
            "errors_and_hits": self._extract_errors_hits(shot_stats, rally_stats),
            "player_ratings": ratings,
            "strategy_insights": strategy.get("insights", []),
            "ai_recommendations": strategy.get("recommendations", {}),
            "strengths_weaknesses": {
                "strengths": strategy.get("strengths", {}),
                "weaknesses": strategy.get("weaknesses", {}),
            },
            "winning_strategies": strategy.get("winning_strategies", []),
            "video_highlights": {
                "match_summary": self.highlight_generator.get_match_summary(),
                "highlights": self.highlight_generator.get_highlights(),
            },
        }

        return report

    def _get_shots_by_type(self, shot_stats: dict) -> dict:
        """Break down shots by type with details"""
        shots_by_type = {}
        for shot in self.shot_detector.shots:
            st = shot.shot_type.value
            if st not in shots_by_type:
                shots_by_type[st] = {"count": 0, "speeds": [], "accuracies": []}
            shots_by_type[st]["count"] += 1
            shots_by_type[st]["speeds"].append(shot.speed_kmh)
            shots_by_type[st]["accuracies"].append(shot.accuracy_score)

        result = {}
        for shot_type, data in shots_by_type.items():
            result[shot_type] = {
                "count": data["count"],
                "avg_speed_kmh": round(np.mean(data["speeds"]), 1) if data["speeds"] else 0,
                "max_speed_kmh": round(max(data["speeds"]), 1) if data["speeds"] else 0,
                "avg_accuracy": round(np.mean(data["accuracies"]), 3) if data["accuracies"] else 0,
            }
        return result

    def _extract_serve_return_stats(self, shot_stats: dict) -> dict:
        """Extract serve and return statistics"""
        serves = [s for s in self.shot_detector.shots if s.shot_type == ShotType.SERVE]

        if not serves:
            return {
                "total_serves": 0,
                "avg_serve_speed_kmh": 0,
                "max_serve_speed_kmh": 0,
                "first_serve_in_pct": 0,
                "per_player": {},
            }

        serve_speeds = [s.speed_kmh for s in serves]

        # Per-player serve stats
        player_serves = {}
        for serve in serves:
            pid = serve.player_id
            if pid not in player_serves:
                player_serves[pid] = {"speeds": [], "count": 0}
            player_serves[pid]["speeds"].append(serve.speed_kmh)
            player_serves[pid]["count"] += 1

        per_player = {}
        for pid, data in player_serves.items():
            per_player[pid] = {
                "total_serves": data["count"],
                "avg_serve_speed_kmh": round(np.mean(data["speeds"]), 1),
                "max_serve_speed_kmh": round(max(data["speeds"]), 1),
            }

        return {
            "total_serves": len(serves),
            "avg_serve_speed_kmh": round(np.mean(serve_speeds), 1),
            "max_serve_speed_kmh": round(max(serve_speeds), 1),
            "first_serve_in_pct": round(len(serves) / max(1, len(serves)) * 100, 1),
            "per_player": per_player,
        }

    def _extract_errors_hits(self, shot_stats: dict, rally_stats: dict) -> dict:
        """Extract errors and hits summary"""
        total_errors = rally_stats.get("errors", 0)
        total_hits = shot_stats.get("total_shots", 0)

        per_player = {}
        for pid, pdata in shot_stats.get("player_stats", {}).items():
            per_player[pid] = {
                "total_hits": pdata.get("total_shots", 0),
                "winners": pdata.get("winners", 0),
                "errors": pdata.get("errors", 0),
                "hit_rate": round(
                    pdata.get("total_shots", 0) / max(1, total_hits) * 100, 1
                ),
            }

        return {
            "total_hits": total_hits,
            "total_errors": total_errors,
            "hit_error_ratio": round(total_hits / max(1, total_errors), 2),
            "per_player": per_player,
        }

    def save_report(self, output_path: str, video_path: Optional[str] = None):
        """Generate and save the report as JSON"""
        report = self.generate_report(video_path)
        
        # Convert numpy types for JSON serialization
        report = self._convert_numpy_types(report)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"comprehensive_stats: Report saved to {output_path}")
        return report

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def print_summary(self):
        """Print a formatted summary to console"""
        report = self.generate_report()

        print("\n" + "="*60)
        print("🏓 AI PADEL COACH - COMPREHENSIVE MATCH ANALYSIS")
        print("="*60)

        print(f"\n📊 MATCH INFO")
        print(f"   Duration: {report['match_info']['duration_seconds']}s")
        print(f"   Frames: {report['match_info']['total_frames_analyzed']}")

        print(f"\n🎾 SHOT STATS")
        print(f"   Total shots: {report['shot_stats']['summary']['total_shots']}")
        print(f"   Distribution: {report['shot_stats']['shot_distribution']}")

        print(f"\n⚡ SHOT SPEED")
        print(f"   Avg: {report['shot_speed']['avg_speed_kmh']} km/h")
        print(f"   Max: {report['shot_speed']['max_speed_kmh']} km/h")

        print(f"\n🎯 SHOT ACCURACY")
        print(f"   Overall: {report['shot_accuracy']['overall_accuracy']:.1%}")

        print(f"\n🔄 RALLY STATS")
        print(f"   Total rallies: {report['rally_stats']['total_rallies']}")
        print(f"   Avg hits/rally: {report['rally_stats']['avg_hits_per_rally']}")
        print(f"   Seconds/point: {report['rally_stats']['seconds_per_point']}")

        print(f"\n📍 COURT COVERAGE")
        for pid, data in report['court_coverage'].get('players', {}).items():
            print(f"   Player {pid}: {data.get('zone_percentages', {})}")

        print(f"\n🏃 MOVEMENT")
        for pid, data in report['movement_positioning'].get('players', {}).items():
            print(f"   Player {pid}: {data.get('total_distance_m', 0)}m, "
                  f"H:{data.get('horizontal_coverage_pct', 0):.0f}%, "
                  f"V:{data.get('vertical_coverage_pct', 0):.0f}%")

        print(f"\n💪 PRESSURE")
        print(f"   Overall rate: {report['pressure_stats'].get('overall_pressure_rate_pct', 0)}%")

        print(f"\n⭐ PLAYER RATINGS")
        for pid, data in report['player_ratings'].items():
            print(f"   Player {pid}: {data.get('overall_rating', 0)}/7 "
                  f"- {data.get('level_description', '')}")

        print(f"\n💡 AI RECOMMENDATIONS")
        for pid, recs in report['ai_recommendations'].items():
            if recs:
                print(f"   Player {pid}:")
                for rec in recs[:3]:
                    print(f"     • [{rec['priority']}] {rec['suggestion'][:80]}")

        print(f"\n🎬 HIGHLIGHTS")
        summary = report['video_highlights']['match_summary']
        print(f"   {summary.get('summary', 'N/A')}")
        print(f"   Total: {summary.get('total_highlights', 0)} highlight moments")

        print("\n" + "="*60)
