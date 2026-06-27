"""
Court Zones Module
Tracks time spent by each player in court zones:
- Volley zone (close to net)
- Transition zone (middle)
- Back zone (close to back wall)
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np


class CourtZone:
    VOLLEY = "volley_zone"
    TRANSITION = "transition_zone"
    BACK = "back_zone"


@dataclass
class ZoneTime:
    """Track frames spent in each zone"""
    volley_frames: int = 0
    transition_frames: int = 0
    back_frames: int = 0

    @property
    def total_frames(self) -> int:
        return self.volley_frames + self.transition_frames + self.back_frames

    def get_percentages(self) -> dict[str, float]:
        total = self.total_frames
        if total == 0:
            return {
                CourtZone.VOLLEY: 0.0,
                CourtZone.TRANSITION: 0.0,
                CourtZone.BACK: 0.0,
            }
        return {
            CourtZone.VOLLEY: round(self.volley_frames / total * 100, 1),
            CourtZone.TRANSITION: round(self.transition_frames / total * 100, 1),
            CourtZone.BACK: round(self.back_frames / total * 100, 1),
        }

    def get_time_seconds(self, fps: float) -> dict[str, float]:
        return {
            CourtZone.VOLLEY: round(self.volley_frames / fps, 2) if fps > 0 else 0,
            CourtZone.TRANSITION: round(self.transition_frames / fps, 2) if fps > 0 else 0,
            CourtZone.BACK: round(self.back_frames / fps, 2) if fps > 0 else 0,
        }


class CourtZoneAnalyzer:
    """
    Analyzes player positioning in court zones.
    
    Padel court (per side):
    - Volley zone: 0 to ~3.3m from net
    - Transition zone: ~3.3m to ~6.6m from net
    - Back zone: ~6.6m to 10m from net (back wall)
    """

    def __init__(
        self, 
        court_length: float = 20.0,
        court_width: float = 10.0,
        volley_zone_depth: float = 3.3,
        transition_zone_depth: float = 6.6,
    ):
        self.court_length = court_length
        self.court_width = court_width
        self.half_court = court_length / 2  # Each side is 10m
        self.volley_zone_depth = volley_zone_depth
        self.transition_zone_depth = transition_zone_depth

        # Per-player zone tracking
        self.player_zones: dict[int, ZoneTime] = {}
        # Zone transition tracking
        self.zone_transitions: dict[int, list[tuple[int, str]]] = {}
        self._last_zones: dict[int, str] = {}

    def reset(self):
        self.player_zones = {}
        self.zone_transitions = {}
        self._last_zones = {}

    def _get_zone(self, player_position: tuple[float, float], player_side: str = "bottom") -> str:
        """
        Determine which zone the player is in based on distance from net.
        
        Args:
            player_position: (x, y) in court coordinates
            player_side: "top" or "bottom" side of court
        """
        y = player_position[1]

        # Calculate distance from net
        if player_side == "bottom":
            # Bottom player: net is at court_length/2, back wall at court_length
            distance_from_net = y - self.half_court
        else:
            # Top player: net is at court_length/2, back wall at 0
            distance_from_net = self.half_court - y

        # Ensure positive distance
        distance_from_net = max(0, distance_from_net)

        if distance_from_net <= self.volley_zone_depth:
            return CourtZone.VOLLEY
        elif distance_from_net <= self.transition_zone_depth:
            return CourtZone.TRANSITION
        else:
            return CourtZone.BACK

    def _get_player_side(self, player_id: int) -> str:
        """
        Determine which side of court the player is on.
        Players 1-2 = bottom (team 1), Players 3-4 = top (team 2)
        """
        return "bottom" if player_id in (1, 2) else "top"

    def update(
        self,
        frame: int,
        player_id: int,
        player_position: tuple[float, float],
    ):
        """Update zone tracking for a player"""
        if player_id not in self.player_zones:
            self.player_zones[player_id] = ZoneTime()
            self.zone_transitions[player_id] = []

        side = self._get_player_side(player_id)
        zone = self._get_zone(player_position, side)

        # Track zone time
        zone_time = self.player_zones[player_id]
        if zone == CourtZone.VOLLEY:
            zone_time.volley_frames += 1
        elif zone == CourtZone.TRANSITION:
            zone_time.transition_frames += 1
        else:
            zone_time.back_frames += 1

        # Track zone transitions
        last_zone = self._last_zones.get(player_id)
        if last_zone is not None and last_zone != zone:
            self.zone_transitions[player_id].append((frame, zone))
        self._last_zones[player_id] = zone

    def get_zone_stats(self, fps: float) -> dict:
        """Get comprehensive zone statistics for all players"""
        stats = {
            "players": {},
            "team_stats": {},
        }

        for player_id, zone_time in self.player_zones.items():
            stats["players"][player_id] = {
                "zone_percentages": zone_time.get_percentages(),
                "zone_time_seconds": zone_time.get_time_seconds(fps),
                "zone_transitions": len(self.zone_transitions.get(player_id, [])),
                "dominant_zone": max(
                    zone_time.get_percentages().items(),
                    key=lambda x: x[1]
                )[0] if zone_time.total_frames > 0 else "unknown",
            }

        # Team stats (players 1-2 vs 3-4)
        for team_name, player_ids in [("team_1", [1, 2]), ("team_2", [3, 4])]:
            team_zones = ZoneTime()
            for pid in player_ids:
                if pid in self.player_zones:
                    pz = self.player_zones[pid]
                    team_zones.volley_frames += pz.volley_frames
                    team_zones.transition_frames += pz.transition_frames
                    team_zones.back_frames += pz.back_frames

            stats["team_stats"][team_name] = {
                "zone_percentages": team_zones.get_percentages(),
                "zone_time_seconds": team_zones.get_time_seconds(fps),
            }

        return stats
