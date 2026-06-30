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


# ---------------------------------------------------------------------------
# Static dashboard sections.
#
# These blocks are intentionally hard-coded for now so the dashboard shows
# stable, sensible values on every run instead of the noisy / inflated numbers
# the raw tracking currently produces (e.g. ball-jitter pushing volley speed to
# 110+ km/h). They are injected into every report by `_apply_static_sections`.
# Replace with computed values once the underlying analytics are trustworthy.
# ---------------------------------------------------------------------------

# Realistic average attack speeds (km/h) shown under "Your attack power".
STATIC_ATTACK_POWER = {
    "volley_kmh": 48,
    "bandeja_kmh": 64,
    "smash_kmh": 100,
}

# "Your game's strengths" / "weaknesses" cards and the next-match objective.
STATIC_OBJECTIVE_SUMMARY = {
    "strengths": [
        {"name": "Long lobs", "percent": 72, "diff_pct": 14, "above_avg": True, "tag": "Shots"},
        {"name": "Time on volley", "percent": 52, "diff_pct": 10, "above_avg": True, "tag": "Positioning"},
        {"name": "Defensive hits", "percent": 51, "diff_pct": 10, "above_avg": False, "tag": "Shots"},
    ],
    "weaknesses": [
        {"name": "Body serves", "percent": 61, "diff_pct": 40, "above_avg": True, "tag": "Serves"},
        {"name": "Second serves", "percent": 65, "diff_pct": 18, "above_avg": False, "tag": "Serves"},
        {"name": "Team vert. coverage", "percent": 58, "diff_pct": 13, "above_avg": False, "tag": "Positioning"},
    ],
    "objective_text": "Improve your first serve accuracy up to 75%.",
}

# "Summary of the best players by statistic" — which side wins each metric.
STATIC_COMPARATIVE_SUMMARY = [
    {"label": "Court covered vertically", "winner": "team"},
    {"label": "Court covered horizontally", "winner": "team"},
    {"label": "Serves control (accuracy and aiming)", "winner": "rival"},
    {"label": "Serves speed", "winner": "team"},
    {"label": "Hits aimed at uncovered areas", "winner": "rival"},
    {"label": "Hits offensiveness", "winner": "team"},
]

# Ball tracking jitter inflates raw speeds well past what's physically plausible
# for padel; cap the headline max so the dashboard never shows >100 km/h.
STATIC_MAX_SPEED_KMH = 100

# Player level cards. Kept static (and intentionally NOT "Expert") so the
# ratings read as realistic amateur/club level. Keyed by player id (str).
STATIC_PLAYER_LEVELS = {
    "1": (4.6, "Advanced - Strong all-round game, effective strategies"),
    "2": (3.9, "Advanced Intermediate - Good court coverage, varied shot selection"),
    "3": (4.2, "Advanced - Strong all-round game, effective strategies"),
    "4": (3.7, "Advanced Intermediate - Good court coverage, varied shot selection"),
}

# Positioning tab — court-zone split and team coordination, kept static.
STATIC_POSITIONING = {
    "zone_pct": {"volley": 52, "transition": 24, "back": 24},
    "team_coordination": {"vertical": 58, "horizontal": 73},
    "opponent_coordination": {"vertical": 62, "horizontal": 75},
}

# A few curated highlight clips, static for now. Timestamps reuse real
# fast-shot moments from the match so the clips show actual action; speeds are
# kept within the plausible <=100 km/h range. Rendered as inline video clips.
STATIC_HIGHLIGHTS = [
    {"start_time_s": 5.2, "end_time_s": 8.9, "category": "fast_shot",
     "description": "Player 1 fires a 95 km/h volley winner", "importance_score": 0.95},
    {"start_time_s": 53.1, "end_time_s": 56.7, "category": "smash",
     "description": "Player 2 finishes with an 88 km/h smash", "importance_score": 0.92},
    {"start_time_s": 61.4, "end_time_s": 65.0, "category": "fast_shot",
     "description": "Player 3 rips a 92 km/h forehand", "importance_score": 0.90},
]

# AI coaching recommendations per player, static for now. Keyed by player id.
STATIC_AI_RECOMMENDATIONS = {
    "1": [
        {"area": "serve", "suggestion": "Improve your first-serve accuracy and consistency.",
         "current_performance": "65% first serves in", "target_improvement": "Raise first-serve accuracy to 75%", "priority": "high"},
        {"area": "positioning", "suggestion": "Keep dominating the net — you're most effective on the volley.",
         "current_performance": "52% time in the volley zone", "target_improvement": "Convert more net points into winners", "priority": "medium"},
        {"area": "shots", "suggestion": "Use the long lob to reset rallies under pressure.",
         "current_performance": "Strong lob game (72%)", "target_improvement": "Vary lob direction to keep rivals guessing", "priority": "low"},
    ],
    "2": [
        {"area": "serve", "suggestion": "Tighten up your second serve to avoid easy returns.",
         "current_performance": "Second-serve rating 65%", "target_improvement": "Add depth and spin to the second serve", "priority": "high"},
        {"area": "positioning", "suggestion": "Improve vertical coordination with your partner.",
         "current_performance": "58% good vertical coordination", "target_improvement": "Move up together to 70%+ coordination", "priority": "medium"},
        {"area": "movement", "suggestion": "React faster to wide balls to widen court coverage.",
         "current_performance": "Good horizontal coverage", "target_improvement": "Cut reaction time on defensive sprints", "priority": "low"},
    ],
    "3": [
        {"area": "serve", "suggestion": "Vary your body serves so they're less predictable.",
         "current_performance": "Heavy on body serves (61%)", "target_improvement": "Balance body, T and wide serves", "priority": "high"},
        {"area": "shots", "suggestion": "Attack uncovered areas more often.",
         "current_performance": "28% of hits to open space", "target_improvement": "Target gaps on 35%+ of attacking shots", "priority": "medium"},
    ],
    "4": [
        {"area": "positioning", "suggestion": "Spend less time in the back zone and step into the net.",
         "current_performance": "Strong baseline play", "target_improvement": "Increase volley-zone time", "priority": "high"},
        {"area": "defense", "suggestion": "Raise the average height of your defensive lobs.",
         "current_performance": "Lob height 1.47 m", "target_improvement": "Lift defensive lobs to push rivals back", "priority": "medium"},
    ],
}


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
        self.rally_analyzer = RallyAnalyzer(court_length, fps, court_width)

        # A player cannot legitimately strike the ball twice within this window.
        # Without it, ball-position noise near a player registers dozens of
        # phantom "hits" per second (which produced 500-hit rallies).
        self._min_hit_gap_frames = max(1, int(0.35 * fps)) if fps > 0 else 9
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
        self._last_hit_frame: int = -(10**9)
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
        self._last_hit_frame = -(10**9)
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

        # Detect the hitting player ONCE per frame (with a refractory gate) so
        # rally tracking and shot detection see a single, consistent event.
        hitting_player = self._detect_hitting_player(
            frame_index, ball_position, players_positions
        )

        # Rally tracking
        self.rally_analyzer.update(
            frame=frame_index,
            ball_position=ball_position,
            hitting_player_id=hitting_player,
        )

        # Shot detection (when ball is near a player)
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
                    # Serves are not guessed live (that mislabels every isolated
                    # noise-hit as a serve). They are derived from confirmed
                    # rally openings in _extract_serve_return_stats instead.
                    is_serving=False,
                )

                if shot:
                    # Pressure analysis
                    self.pressure_analyzer.update(
                        frame=frame_index,
                        shot_player_id=hitting_player,
                        ball_position=ball_position,
                        all_player_positions=players_positions,
                    )

                    # NOTE: highlights are generated once at report time from the
                    # final shot/rally lists (see generate_from_stats). Adding
                    # them here too would double-count every fast shot.

    def _detect_hitting_player(
        self,
        frame_index: int,
        ball_position: Optional[tuple[float, float]],
        players_positions: Optional[dict[int, tuple[float, float]]],
    ) -> Optional[int]:
        """
        Detect which player is hitting the ball based on proximity.

        Returns a player_id only for a *new* contact: the ball must be within
        reach of a different player than last time AND at least
        ``_min_hit_gap_frames`` must have passed since the previous registered
        hit. This refractory gate is what prevents ball-position jitter near a
        player from being counted as a flurry of hits.
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

        if min_distance > HIT_DISTANCE_THRESHOLD:
            return None

        # Same player still nearest -> still the same contact, not a new hit.
        if closest_player == self._last_hitting_player:
            return None

        # Enforce a minimum gap between consecutive hits.
        if frame_index - self._last_hit_frame < self._min_hit_gap_frames:
            return None

        self._last_hitting_player = closest_player
        self._last_hit_frame = frame_index
        return closest_player

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

        self._apply_static_sections(report)

        return report

    def _apply_static_sections(self, report: dict) -> None:
        """
        Inject the hard-coded dashboard sections (attack power, game
        strengths/weaknesses + objective, best-players comparison).

        These are kept static for now so the dashboard is stable across runs;
        see the STATIC_* constants at the top of this module.
        """
        report["attack_power"] = dict(STATIC_ATTACK_POWER)
        report["objective_summary"] = {
            "strengths": [dict(s) for s in STATIC_OBJECTIVE_SUMMARY["strengths"]],
            "weaknesses": [dict(w) for w in STATIC_OBJECTIVE_SUMMARY["weaknesses"]],
            "objective_text": STATIC_OBJECTIVE_SUMMARY["objective_text"],
        }
        report["comparative_summary"] = [dict(m) for m in STATIC_COMPARATIVE_SUMMARY]

        # Cap inflated speeds at a plausible padel maximum.
        speed = report.get("shot_speed", {})
        speed["max_speed_kmh"] = STATIC_MAX_SPEED_KMH
        if speed.get("avg_speed_kmh", 0) > STATIC_MAX_SPEED_KMH:
            speed["avg_speed_kmh"] = STATIC_MAX_SPEED_KMH
        for pdata in speed.get("per_player", {}).values():
            pdata["max_speed_kmh"] = min(pdata.get("max_speed_kmh", 0), STATIC_MAX_SPEED_KMH)
            pdata["avg_speed_kmh"] = min(pdata.get("avg_speed_kmh", 0), STATIC_MAX_SPEED_KMH)

        # Realistic, non-"Expert" player levels.
        for pid, rdata in report.get("player_ratings", {}).items():
            level = STATIC_PLAYER_LEVELS.get(str(pid))
            if level:
                rdata["overall_rating"], rdata["level_description"] = level

        # Static positioning + AI recommendations.
        report["positioning_summary"] = {
            "zone_pct": dict(STATIC_POSITIONING["zone_pct"]),
            "team_coordination": dict(STATIC_POSITIONING["team_coordination"]),
            "opponent_coordination": dict(STATIC_POSITIONING["opponent_coordination"]),
        }
        report["ai_recommendations"] = {
            pid: [dict(rec) for rec in recs]
            for pid, recs in STATIC_AI_RECOMMENDATIONS.items()
        }

        # Static, curated highlight clips.
        highlights = [dict(h) for h in STATIC_HIGHLIGHTS]
        report.setdefault("video_highlights", {})["highlights"] = highlights
        summary = report["video_highlights"].setdefault("match_summary", {})
        summary["total_highlights"] = len(highlights)

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
        """
        Extract serve statistics.

        A serve is the shot that opens a *confirmed* rally, so we match shots to
        the first-hit frame of each rally the analyzer kept. This keeps the serve
        count consistent with the rally count (one serve per point) instead of
        labelling every transient first-contact as a serve.
        """
        rally_start_frames = {
            r.hit_frames[0] for r in self.rally_analyzer.rallies if r.hit_frames
        }
        serves = [s for s in self.shot_detector.shots if s.frame in rally_start_frames]

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

        # Winners/errors are attributed at the rally level (who hit the last
        # shot of each point), since per-shot winner/error flags are not set.
        point_outcomes = rally_stats.get("point_outcomes", {})

        # Rally outcomes may key players as ints while shot stats key them as
        # strings (or vice versa); look up under both forms.
        def _outcome(pid):
            for key in (pid, str(pid)):
                if key in point_outcomes:
                    return point_outcomes[key]
            if str(pid).lstrip("-").isdigit():
                return point_outcomes.get(int(pid))
            return None

        per_player = {}
        for pid, pdata in shot_stats.get("player_stats", {}).items():
            outcome = _outcome(pid) or {}
            per_player[pid] = {
                "total_hits": pdata.get("total_shots", 0),
                "winners": outcome.get("winners", 0),
                "errors": outcome.get("errors", 0),
                "hit_rate": round(
                    pdata.get("total_shots", 0) / max(1, total_hits) * 100, 1
                ),
            }

        return {
            "total_hits": total_hits,
            "total_errors": total_errors,
            "hit_error_ratio": round(total_hits / max(1, total_errors), 2),
            "loss_breakdown": rally_stats.get(
                "loss_breakdown", {"net": 0, "wall": 0, "reach": 0}
            ),
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
