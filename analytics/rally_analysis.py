"""
Rally Analysis Module
Tracks rallies: total count, rally length, seconds per point, average hits per point
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Rally:
    start_frame: int
    end_frame: Optional[int] = None
    hits: int = 0
    hit_frames: list = field(default_factory=list)
    winner_player_id: Optional[int] = None
    is_error: bool = False
    error_player_id: Optional[int] = None

    @property
    def duration_frames(self) -> int:
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame

    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames / fps if fps > 0 else 0.0

    def serialize(self, fps: float) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "hits": self.hits,
            "duration_seconds": round(self.duration_seconds(fps), 2),
            "winner_player_id": self.winner_player_id,
            "is_error": self.is_error,
            "error_player_id": self.error_player_id,
        }


class RallyAnalyzer:
    """
    Analyzes rallies based on ball crossings over the net.
    A rally starts with a serve and ends when:
    - Ball goes out of play (no detection for N frames)
    - Ball hits the ground twice on same side
    - Error detected
    """

    # Thresholds
    BALL_LOST_FRAMES = 30  # frames without ball detection = rally end
    NET_POSITION_RATIO = 0.5  # net is at 50% of court length
    MIN_RALLY_HITS = 1  # minimum hits to count as rally

    def __init__(self, court_length: float = 20.0, fps: float = 30.0):
        self.court_length = court_length
        self.fps = fps
        self.rallies: list[Rally] = []
        self.current_rally: Optional[Rally] = None
        self._last_ball_frame: int = 0
        self._last_ball_side: Optional[str] = None  # "top" or "bottom"
        self._frames_without_ball: int = 0
        self._ball_crossed_net: bool = False

    def reset(self):
        self.rallies = []
        self.current_rally = None
        self._last_ball_frame = 0
        self._last_ball_side = None
        self._frames_without_ball = 0
        self._ball_crossed_net = False

    def _get_ball_side(self, ball_y: float) -> str:
        """Determine which side of the net the ball is on"""
        net_y = self.court_length * self.NET_POSITION_RATIO
        return "top" if ball_y < net_y else "bottom"

    def update(
        self,
        frame: int,
        ball_position: Optional[tuple[float, float]],
        hitting_player_id: Optional[int] = None,
    ):
        """
        Update rally state for each frame.
        
        Args:
            frame: current frame number
            ball_position: ball (x, y) in court coordinates, None if not detected
            hitting_player_id: player ID if a hit was detected this frame
        """
        if ball_position is None:
            self._frames_without_ball += 1
            if (self._frames_without_ball >= self.BALL_LOST_FRAMES 
                and self.current_rally is not None):
                self._end_rally(frame)
            return

        self._frames_without_ball = 0
        self._last_ball_frame = frame

        # Determine ball side
        current_side = self._get_ball_side(ball_position[1])

        # Detect net crossing (ball changed side)
        if self._last_ball_side is not None and current_side != self._last_ball_side:
            self._ball_crossed_net = True
            if self.current_rally is not None:
                self.current_rally.hits += 1
                self.current_rally.hit_frames.append(frame)

        self._last_ball_side = current_side

        # If a hit is detected and no rally is active, start one
        if hitting_player_id is not None:
            if self.current_rally is None:
                self._start_rally(frame)
            self.current_rally.hits += 1
            self.current_rally.hit_frames.append(frame)

    def _start_rally(self, frame: int):
        """Start a new rally"""
        self.current_rally = Rally(start_frame=frame)

    def _end_rally(self, frame: int):
        """End the current rally"""
        if self.current_rally is not None:
            self.current_rally.end_frame = frame
            if self.current_rally.hits >= self.MIN_RALLY_HITS:
                self.rallies.append(self.current_rally)
            self.current_rally = None
            self._ball_crossed_net = False

    def force_end_rally(self, frame: int, winner_id: Optional[int] = None, 
                        is_error: bool = False, error_player_id: Optional[int] = None):
        """Force end a rally (e.g., when a point is won)"""
        if self.current_rally is not None:
            self.current_rally.winner_player_id = winner_id
            self.current_rally.is_error = is_error
            self.current_rally.error_player_id = error_player_id
            self._end_rally(frame)

    def get_rally_stats(self) -> dict:
        """Get comprehensive rally statistics"""
        if not self.rallies:
            return self._empty_stats()

        total_rallies = len(self.rallies)
        durations = [r.duration_seconds(self.fps) for r in self.rallies]
        hits_per_rally = [r.hits for r in self.rallies]

        return {
            "total_rallies": total_rallies,
            "avg_rally_length_seconds": round(np.mean(durations), 2) if durations else 0,
            "max_rally_length_seconds": round(max(durations), 2) if durations else 0,
            "min_rally_length_seconds": round(min(durations), 2) if durations else 0,
            "avg_hits_per_rally": round(np.mean(hits_per_rally), 1) if hits_per_rally else 0,
            "max_hits_per_rally": max(hits_per_rally) if hits_per_rally else 0,
            "total_hits": sum(hits_per_rally),
            "seconds_per_point": round(np.mean(durations), 2) if durations else 0,
            "errors": sum(1 for r in self.rallies if r.is_error),
            "rally_details": [r.serialize(self.fps) for r in self.rallies],
        }

    def _empty_stats(self) -> dict:
        return {
            "total_rallies": 0,
            "avg_rally_length_seconds": 0,
            "max_rally_length_seconds": 0,
            "min_rally_length_seconds": 0,
            "avg_hits_per_rally": 0,
            "max_hits_per_rally": 0,
            "total_hits": 0,
            "seconds_per_point": 0,
            "errors": 0,
            "rally_details": [],
        }
