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
    # How the point ended: "net", "wall", or "reach" (opponent couldn't reach).
    loss_reason: Optional[str] = None

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
            "loss_reason": self.loss_reason,
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
    MIN_RALLY_HITS = 2  # a real rally needs at least a serve + a return
    # Seconds the ball can stay in view without anyone hitting it before the
    # point is considered over. TrackNet keeps "seeing" the ball between points,
    # so without this a whole game collapsed into one 300s+ rally.
    HIT_TIMEOUT_SECONDS = 2.0
    # Heuristic bands (meters) for classifying how a point ended.
    NET_BAND = 1.2  # ball died within this distance of the net line -> net error
    WALL_BAND = 0.7  # ball ended within this distance of a wall -> direct-to-wall
    # How many recent ball positions to remember for end-of-rally classification.
    BALL_TRAIL_LEN = 15

    def __init__(self, court_length: float = 20.0, fps: float = 30.0,
                 court_width: float = 10.0):
        self.court_length = court_length
        self.court_width = court_width
        self.fps = fps
        self.rallies: list[Rally] = []
        self.current_rally: Optional[Rally] = None
        self._last_ball_frame: int = 0
        self._last_ball_side: Optional[str] = None  # "top" or "bottom"
        self._frames_without_ball: int = 0
        self._ball_crossed_net: bool = False
        self._last_hitter: Optional[int] = None
        self._last_hit_frame: Optional[int] = None
        self._ball_trail: list[tuple[float, float]] = []
        self._hit_timeout_frames = max(1, int(self.HIT_TIMEOUT_SECONDS * fps)) if fps > 0 else 50

    def reset(self):
        self.rallies = []
        self.current_rally = None
        self._last_ball_frame = 0
        self._last_ball_side = None
        self._frames_without_ball = 0
        self._ball_crossed_net = False
        self._last_hitter = None
        self._last_hit_frame = None
        self._ball_trail = []

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

        # Remember the recent ball path so we can classify how the point ended.
        self._ball_trail.append((ball_position[0], ball_position[1]))
        if len(self._ball_trail) > self.BALL_TRAIL_LEN:
            self._ball_trail = self._ball_trail[-self.BALL_TRAIL_LEN:]

        # Determine ball side
        current_side = self._get_ball_side(ball_position[1])

        # Track net crossings for end-of-point context only. These are NOT
        # counted as hits: the ball oscillates across the net line due to
        # projection noise, and counting each crossing inflated rallies to
        # hundreds of "hits". Only real player contacts (below) count.
        if self._last_ball_side is not None and current_side != self._last_ball_side:
            self._ball_crossed_net = True

        self._last_ball_side = current_side

        # A genuine player contact (already de-duplicated upstream with a
        # refractory gate) is the only thing that counts as a hit.
        if hitting_player_id is not None:
            if self.current_rally is None:
                self._start_rally(frame)
            self.current_rally.hits += 1
            self.current_rally.hit_frames.append(frame)
            self._last_hitter = hitting_player_id
            self._last_hit_frame = frame
        elif (
            self.current_rally is not None
            and self._last_hit_frame is not None
            and frame - self._last_hit_frame >= self._hit_timeout_frames
        ):
            # Ball still tracked but nobody has hit it for a while -> point over.
            self._end_rally(self._last_hit_frame)

    def _start_rally(self, frame: int):
        """Start a new rally"""
        self.current_rally = Rally(start_frame=frame)
        self._last_hitter = None
        self._last_hit_frame = frame
        self._ball_trail = []

    def _end_rally(self, frame: int):
        """End the current rally"""
        if self.current_rally is not None:
            self.current_rally.end_frame = frame
            if self.current_rally.hits >= self.MIN_RALLY_HITS:
                self._classify_ending(self.current_rally)
                self.rallies.append(self.current_rally)
            self.current_rally = None
            self._ball_crossed_net = False

    def _classify_ending(self, rally: "Rally"):
        """
        Heuristically classify how a point ended from the last known ball
        position and the player who hit the final shot.

        Reasons:
        - "net":   ball died near the net line  -> error by the last hitter.
        - "wall":  ball ended jammed against a wall -> direct-to-wall error
                   by the last hitter.
        - "reach": ball ended in open court -> treated as a winner for the
                   last hitter (the opponent could not reach it).

        Approximate by design: it infers cause from 2D court position only,
        with no ground-truth bounce or fault calls.
        """
        if not self._ball_trail:
            return

        bx, by = self._ball_trail[-1]
        hitter = self._last_hitter
        net_y = self.court_length * self.NET_POSITION_RATIO

        near_side_wall = bx <= self.WALL_BAND or bx >= self.court_width - self.WALL_BAND
        near_back_wall = by <= self.WALL_BAND or by >= self.court_length - self.WALL_BAND

        if abs(by - net_y) <= self.NET_BAND:
            rally.loss_reason = "net"
            rally.is_error = True
            rally.error_player_id = hitter
        elif near_side_wall or near_back_wall:
            rally.loss_reason = "wall"
            rally.is_error = True
            rally.error_player_id = hitter
        else:
            rally.loss_reason = "reach"
            rally.winner_player_id = hitter

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
            "loss_breakdown": self._loss_breakdown(),
            "point_outcomes": self._point_outcomes(),
            "rally_details": [r.serialize(self.fps) for r in self.rallies],
        }

    def _loss_breakdown(self) -> dict:
        """Count how points ended, across all classified rallies."""
        breakdown = {"net": 0, "wall": 0, "reach": 0}
        for r in self.rallies:
            if r.loss_reason in breakdown:
                breakdown[r.loss_reason] += 1
        return breakdown

    def _point_outcomes(self) -> dict:
        """Per-player winners (points won) and errors (points lost), keyed by player id."""
        outcomes: dict[int, dict] = {}
        for r in self.rallies:
            if r.winner_player_id is not None:
                o = outcomes.setdefault(r.winner_player_id, {"winners": 0, "errors": 0})
                o["winners"] += 1
            if r.is_error and r.error_player_id is not None:
                o = outcomes.setdefault(r.error_player_id, {"winners": 0, "errors": 0})
                o["errors"] += 1
        return outcomes

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
            "loss_breakdown": {"net": 0, "wall": 0, "reach": 0},
            "point_outcomes": {},
            "rally_details": [],
        }
