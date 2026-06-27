"""
Shot Detection Module
Classifies shots into: forehand, backhand, smash, volley, lob
Uses player pose keypoints + ball trajectory + player court position
"""

from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math


class ShotType(Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SMASH = "smash"
    VOLLEY = "volley"
    LOB = "lob"
    SERVE = "serve"
    UNKNOWN = "unknown"


@dataclass
class Shot:
    frame: int
    player_id: int
    shot_type: ShotType
    speed_ms: float  # meters per second
    ball_position: tuple[float, float]
    player_position: tuple[float, float]
    is_winner: bool = False
    is_error: bool = False
    placement_zone: Optional[str] = None  # e.g., "deep_left", "short_right"
    accuracy_score: float = 0.0  # 0-1 score

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6

    def serialize(self) -> dict:
        return {
            "frame": self.frame,
            "player_id": self.player_id,
            "shot_type": self.shot_type.value,
            "speed_ms": round(self.speed_ms, 2),
            "speed_kmh": round(self.speed_kmh, 2),
            "ball_position": self.ball_position,
            "player_position": self.player_position,
            "is_winner": self.is_winner,
            "is_error": self.is_error,
            "placement_zone": self.placement_zone,
            "accuracy_score": round(self.accuracy_score, 3),
        }


class ShotDetector:
    """
    Detects and classifies shots based on:
    - Player pose keypoints (arm position, body orientation)
    - Ball trajectory and speed
    - Player position on court (volley zone vs back zone)
    - Ball height trajectory (for lobs/smashes)
    """

    # Court zones (distance from net in meters, padel court is 10m each side)
    VOLLEY_ZONE_MAX = 3.0  # 0-3m from net
    TRANSITION_ZONE_MAX = 6.0  # 3-6m from net
    BACK_ZONE_MAX = 10.0  # 6-10m from net

    # Thresholds for shot classification
    SMASH_HEIGHT_THRESHOLD = 0.7  # arm above 70% of body height
    LOB_BALL_HEIGHT_THRESHOLD = 3.0  # meters above net level
    VOLLEY_DISTANCE_FROM_NET = 3.5  # meters
    MIN_BALL_SPEED_FOR_SHOT = 2.0  # m/s minimum to count as shot

    def __init__(self, court_length: float = 20.0, court_width: float = 10.0):
        self.court_length = court_length
        self.court_width = court_width
        self.shots: list[Shot] = []
        self._prev_ball_positions: list[tuple[int, float, float]] = []
        self._prev_ball_side: Optional[str] = None  # "top" or "bottom"

    def reset(self):
        self.shots = []
        self._prev_ball_positions = []
        self._prev_ball_side = None

    def _get_arm_elevation(
        self, 
        player_keypoints: dict[str, tuple[float, float]],
    ) -> float:
        """
        Calculate arm elevation relative to body height.
        Returns value 0-1 (1 = arm fully above head)
        """
        try:
            head = player_keypoints.get("head")
            right_hand = player_keypoints.get("right_hand")
            left_hand = player_keypoints.get("left_hand")
            left_foot = player_keypoints.get("left_foot")
            right_foot = player_keypoints.get("right_foot")

            if not all([head, left_foot, right_foot]):
                return 0.0

            body_height = abs(head[1] - max(left_foot[1], right_foot[1]))
            if body_height == 0:
                return 0.0

            max_hand_height = 0.0
            if right_hand:
                max_hand_height = max(max_hand_height, head[1] - right_hand[1])
            if left_hand:
                max_hand_height = max(max_hand_height, head[1] - left_hand[1])

            return max(0.0, min(1.0, max_hand_height / body_height))
        except (TypeError, KeyError):
            return 0.0

    def _get_dominant_hand_side(
        self,
        player_keypoints: dict[str, tuple[float, float]],
        ball_position: tuple[float, float],
    ) -> Literal["forehand", "backhand"]:
        """
        Determine if shot is forehand or backhand based on 
        which hand is closer to ball and body orientation.
        """
        try:
            right_hand = player_keypoints.get("right_hand")
            left_hand = player_keypoints.get("left_hand")
            right_shoulder = player_keypoints.get("right_shoulder")
            left_shoulder = player_keypoints.get("left_shoulder")
            torso = player_keypoints.get("torso")

            if not torso or not ball_position:
                return "forehand"

            # Determine body facing direction from shoulders
            if right_shoulder and left_shoulder:
                body_facing_x = right_shoulder[0] - left_shoulder[0]
            else:
                body_facing_x = 0

            # Ball relative to torso
            ball_relative_x = ball_position[0] - torso[0]

            # If ball is on the same side as the body is facing -> forehand
            # If ball is on opposite side -> backhand
            if (body_facing_x > 0 and ball_relative_x > 0) or \
               (body_facing_x < 0 and ball_relative_x < 0):
                return "forehand"
            else:
                return "backhand"

        except (TypeError, KeyError):
            return "forehand"

    def _is_in_volley_zone(
        self, 
        player_position: tuple[float, float],
    ) -> bool:
        """Check if player is in the volley zone (close to net)"""
        # y-coordinate represents distance from net (0 = net, 10 = back wall)
        distance_from_net = abs(player_position[1] - self.court_length / 2)
        # Adjust for which side of court player is on
        if player_position[1] < self.court_length / 2:
            distance_from_net = player_position[1]
        else:
            distance_from_net = self.court_length - player_position[1]
        return distance_from_net <= self.VOLLEY_DISTANCE_FROM_NET

    def _calculate_ball_speed(
        self,
        ball_positions: list[tuple[int, float, float]],
        fps: float,
        pixels_per_meter: float = 1.0,
    ) -> float:
        """Calculate ball speed from recent positions in m/s"""
        if len(ball_positions) < 2:
            return 0.0

        p1 = ball_positions[-2]
        p2 = ball_positions[-1]

        frame_diff = p2[0] - p1[0]
        if frame_diff == 0:
            return 0.0

        dx = (p2[1] - p1[1]) / pixels_per_meter
        dy = (p2[2] - p1[2]) / pixels_per_meter
        distance = math.sqrt(dx**2 + dy**2)

        time_elapsed = frame_diff / fps
        return distance / time_elapsed if time_elapsed > 0 else 0.0

    def _estimate_ball_height_change(
        self,
        ball_positions: list[tuple[int, float, float]],
    ) -> float:
        """
        Estimate if ball is going up (positive) or down (negative)
        using y-coordinate changes (in image space, up = negative y)
        """
        if len(ball_positions) < 3:
            return 0.0

        recent = ball_positions[-3:]
        y_changes = [recent[i+1][2] - recent[i][2] for i in range(len(recent)-1)]
        return -np.mean(y_changes)  # negative because image y is inverted

    def _get_placement_zone(
        self,
        ball_position: tuple[float, float],
    ) -> str:
        """Determine where on court the ball lands"""
        x, y = ball_position
        half_width = self.court_width / 2

        # Horizontal: left, center, right
        if x < half_width * 0.33:
            h_zone = "left"
        elif x < half_width * 0.66:
            h_zone = "center"
        else:
            h_zone = "right"

        # Depth: short, mid, deep
        if y < self.court_length * 0.33:
            d_zone = "short"
        elif y < self.court_length * 0.66:
            d_zone = "mid"
        else:
            d_zone = "deep"

        return f"{d_zone}_{h_zone}"

    def detect_shot(
        self,
        frame: int,
        player_id: int,
        player_keypoints: Optional[dict[str, tuple[float, float]]],
        player_position: tuple[float, float],
        ball_position: tuple[float, float],
        ball_positions_history: list[tuple[int, float, float]],
        fps: float,
        pixels_per_meter: float = 1.0,
        is_serving: bool = False,
    ) -> Optional[Shot]:
        """
        Detect and classify a shot based on all available tracking data.
        
        Returns Shot object if a shot is detected, None otherwise.
        """
        # Calculate ball speed
        speed = self._calculate_ball_speed(ball_positions_history, fps, pixels_per_meter)

        if speed < self.MIN_BALL_SPEED_FOR_SHOT:
            return None

        # Determine shot type
        shot_type = ShotType.UNKNOWN

        if is_serving:
            shot_type = ShotType.SERVE
        elif self._is_in_volley_zone(player_position):
            # Player is near net -> likely volley
            shot_type = ShotType.VOLLEY
        elif player_keypoints:
            arm_elevation = self._get_arm_elevation(player_keypoints)
            ball_height_change = self._estimate_ball_height_change(ball_positions_history)

            if arm_elevation > self.SMASH_HEIGHT_THRESHOLD and ball_height_change < -1.0:
                # Arm high + ball coming down = smash
                shot_type = ShotType.SMASH
            elif ball_height_change > 2.0:
                # Ball going significantly upward = lob
                shot_type = ShotType.LOB
            else:
                # Determine forehand vs backhand
                hand_side = self._get_dominant_hand_side(
                    player_keypoints, ball_position
                )
                shot_type = (
                    ShotType.FOREHAND if hand_side == "forehand" 
                    else ShotType.BACKHAND
                )
        else:
            # No keypoints available, classify by position
            ball_height_change = self._estimate_ball_height_change(ball_positions_history)
            if ball_height_change > 2.0:
                shot_type = ShotType.LOB
            else:
                shot_type = ShotType.FOREHAND  # default

        # Determine placement
        placement = self._get_placement_zone(ball_position)

        # Calculate accuracy (based on how close to lines/corners)
        accuracy = self._calculate_accuracy(ball_position)

        shot = Shot(
            frame=frame,
            player_id=player_id,
            shot_type=shot_type,
            speed_ms=speed,
            ball_position=ball_position,
            player_position=player_position,
            placement_zone=placement,
            accuracy_score=accuracy,
        )

        self.shots.append(shot)
        return shot

    def _calculate_accuracy(self, ball_position: tuple[float, float]) -> float:
        """
        Calculate shot accuracy based on placement quality.
        Shots closer to lines/corners score higher.
        """
        x, y = ball_position
        half_width = self.court_width / 2

        # Distance to nearest sideline
        dist_to_sideline = min(x, half_width - x) if half_width > 0 else 0
        # Distance to nearest baseline
        dist_to_baseline = min(y, self.court_length - y) if self.court_length > 0 else 0

        # Closer to lines = higher accuracy (inverted and normalized)
        max_dist = math.sqrt((half_width/2)**2 + (self.court_length/2)**2)
        if max_dist == 0:
            return 0.5

        min_dist_to_line = min(dist_to_sideline, dist_to_baseline)
        accuracy = 1.0 - (min_dist_to_line / (half_width / 2))
        return max(0.0, min(1.0, accuracy))

    def get_shot_stats(self) -> dict:
        """Get comprehensive shot statistics"""
        if not self.shots:
            return self._empty_stats()

        total = len(self.shots)
        by_type = {}
        by_player = {}
        speeds = []
        accuracies = []

        for shot in self.shots:
            # By type
            st = shot.shot_type.value
            by_type[st] = by_type.get(st, 0) + 1

            # By player
            pid = shot.player_id
            if pid not in by_player:
                by_player[pid] = {"total": 0, "types": {}, "speeds": [], "accuracies": []}
            by_player[pid]["total"] += 1
            by_player[pid]["types"][st] = by_player[pid]["types"].get(st, 0) + 1
            by_player[pid]["speeds"].append(shot.speed_kmh)
            by_player[pid]["accuracies"].append(shot.accuracy_score)

            speeds.append(shot.speed_kmh)
            accuracies.append(shot.accuracy_score)

        # Compute per-player stats
        player_stats = {}
        for pid, pdata in by_player.items():
            player_stats[pid] = {
                "total_shots": pdata["total"],
                "shot_types": pdata["types"],
                "avg_speed_kmh": round(np.mean(pdata["speeds"]), 1) if pdata["speeds"] else 0,
                "max_speed_kmh": round(max(pdata["speeds"]), 1) if pdata["speeds"] else 0,
                "avg_accuracy": round(np.mean(pdata["accuracies"]), 3) if pdata["accuracies"] else 0,
                "winners": sum(1 for s in self.shots if s.player_id == pid and s.is_winner),
                "errors": sum(1 for s in self.shots if s.player_id == pid and s.is_error),
            }

        return {
            "total_shots": total,
            "shot_distribution": by_type,
            "avg_speed_kmh": round(np.mean(speeds), 1) if speeds else 0,
            "max_speed_kmh": round(max(speeds), 1) if speeds else 0,
            "avg_accuracy": round(np.mean(accuracies), 3) if accuracies else 0,
            "player_stats": player_stats,
            "shots_detail": [s.serialize() for s in self.shots],
        }

    def _empty_stats(self) -> dict:
        return {
            "total_shots": 0,
            "shot_distribution": {},
            "avg_speed_kmh": 0,
            "max_speed_kmh": 0,
            "avg_accuracy": 0,
            "player_stats": {},
            "shots_detail": [],
        }
