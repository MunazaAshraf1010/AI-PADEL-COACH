"""
Pressure Stats Module
Measures shot pressure: % of shots forcing opponent to move >2 meters
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class PressureEvent:
    frame: int
    shooting_player_id: int
    receiving_player_id: int
    opponent_distance_to_move: float  # meters
    is_pressure_shot: bool  # True if > 2m movement required


class PressureAnalyzer:
    """
    Analyzes shot pressure:
    - % of shots that force opponent to move > 2 meters
    - Average displacement forced on opponents
    - Pressure index per player
    """

    PRESSURE_THRESHOLD_M = 2.0  # meters opponent must move to be "pressure"

    def __init__(self, court_length: float = 20.0, court_width: float = 10.0):
        self.court_length = court_length
        self.court_width = court_width
        self.pressure_events: list[PressureEvent] = []
        self._prev_opponent_positions: dict[int, tuple[float, float]] = {}

    def reset(self):
        self.pressure_events = []
        self._prev_opponent_positions = {}

    def _get_opponent_ids(self, player_id: int) -> list[int]:
        """Get opponent player IDs (team 1: 1,2 vs team 2: 3,4)"""
        if player_id in (1, 2):
            return [3, 4]
        else:
            return [1, 2]

    def update(
        self,
        frame: int,
        shot_player_id: int,
        ball_position: tuple[float, float],
        all_player_positions: dict[int, tuple[float, float]],
    ):
        """
        Update pressure tracking when a shot is detected.
        
        Args:
            frame: current frame
            shot_player_id: player who hit the shot
            ball_position: where ball is going
            all_player_positions: current positions of all players
        """
        opponent_ids = self._get_opponent_ids(shot_player_id)

        for opp_id in opponent_ids:
            if opp_id not in all_player_positions:
                continue

            opp_position = all_player_positions[opp_id]

            # How far the opponent was actually forced to move since the
            # previous shot. The absolute ball-to-opponent distance is NOT a
            # pressure signal: opponents stand on the far side of the court, so
            # that distance is almost always > 2m and reported ~95% "pressure"
            # for everyone. Real pressure is displacement the shot forces.
            prev_position = self._prev_opponent_positions.get(opp_id)
            if prev_position is None:
                # No baseline yet (first shot we see this opponent) -> skip,
                # rather than inventing a pressure event.
                continue

            dx = opp_position[0] - prev_position[0]
            dy = opp_position[1] - prev_position[1]
            displacement = math.sqrt(dx**2 + dy**2)

            is_pressure = displacement >= self.PRESSURE_THRESHOLD_M

            event = PressureEvent(
                frame=frame,
                shooting_player_id=shot_player_id,
                receiving_player_id=opp_id,
                opponent_distance_to_move=displacement,
                is_pressure_shot=is_pressure,
            )
            self.pressure_events.append(event)

        # Update previous positions
        for pid, pos in all_player_positions.items():
            self._prev_opponent_positions[pid] = pos

    def get_pressure_stats(self) -> dict:
        """Get comprehensive pressure statistics"""
        if not self.pressure_events:
            return self._empty_stats()

        total_events = len(self.pressure_events)
        pressure_shots = sum(1 for e in self.pressure_events if e.is_pressure_shot)
        distances = [e.opponent_distance_to_move for e in self.pressure_events]

        # Per-player pressure applied
        player_pressure_applied: dict[int, list[PressureEvent]] = {}
        player_pressure_received: dict[int, list[PressureEvent]] = {}

        for event in self.pressure_events:
            # Pressure applied by shooter
            if event.shooting_player_id not in player_pressure_applied:
                player_pressure_applied[event.shooting_player_id] = []
            player_pressure_applied[event.shooting_player_id].append(event)

            # Pressure received by opponent
            if event.receiving_player_id not in player_pressure_received:
                player_pressure_received[event.receiving_player_id] = []
            player_pressure_received[event.receiving_player_id].append(event)

        # Build per-player stats
        players_stats = {}
        all_player_ids = set(
            list(player_pressure_applied.keys()) + list(player_pressure_received.keys())
        )

        for pid in all_player_ids:
            applied = player_pressure_applied.get(pid, [])
            received = player_pressure_received.get(pid, [])

            applied_pressure = sum(1 for e in applied if e.is_pressure_shot)
            received_pressure = sum(1 for e in received if e.is_pressure_shot)

            players_stats[pid] = {
                "pressure_shots_applied": applied_pressure,
                "total_shots": len(applied),
                "pressure_rate_applied_pct": round(
                    applied_pressure / len(applied) * 100, 1
                ) if applied else 0,
                "avg_distance_forced_m": round(
                    np.mean([e.opponent_distance_to_move for e in applied]), 2
                ) if applied else 0,
                "pressure_shots_received": received_pressure,
                "pressure_rate_received_pct": round(
                    received_pressure / len(received) * 100, 1
                ) if received else 0,
                "avg_distance_forced_to_move_m": round(
                    np.mean([e.opponent_distance_to_move for e in received]), 2
                ) if received else 0,
            }

        return {
            "total_pressure_shots": pressure_shots,
            "total_shots_analyzed": total_events,
            "overall_pressure_rate_pct": round(pressure_shots / total_events * 100, 1),
            "avg_opponent_displacement_m": round(np.mean(distances), 2),
            "max_opponent_displacement_m": round(max(distances), 2),
            "player_stats": players_stats,
        }

    def _empty_stats(self) -> dict:
        return {
            "total_pressure_shots": 0,
            "total_shots_analyzed": 0,
            "overall_pressure_rate_pct": 0,
            "avg_opponent_displacement_m": 0,
            "max_opponent_displacement_m": 0,
            "player_stats": {},
        }
