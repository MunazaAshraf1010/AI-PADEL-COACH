"""
Movement Analysis Module
Tracks movement patterns, horizontal/vertical coverage, team positioning dynamics
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class MovementFrame:
    frame: int
    position: tuple[float, float]


class MovementAnalyzer:
    """
    Analyzes player movement patterns:
    - Total distance covered
    - Horizontal coverage (side-to-side movement)
    - Vertical coverage (front-to-back movement)
    - Movement speed and acceleration
    - Team positioning (doubles formation analysis)
    - Heat map data
    """

    def __init__(
        self, 
        court_length: float = 20.0,
        court_width: float = 10.0,
        fps: float = 30.0,
    ):
        self.court_length = court_length
        self.court_width = court_width
        self.fps = fps

        # Per-player tracking
        self.player_positions: dict[int, list[MovementFrame]] = {}
        # Heat map grid (divide court into cells)
        self.heatmap_grid_size = 10  # 10x10 grid
        self.player_heatmaps: dict[int, np.ndarray] = {}

    def reset(self):
        self.player_positions = {}
        self.player_heatmaps = {}

    def update(
        self,
        frame: int,
        player_id: int,
        position: tuple[float, float],
    ):
        """Update position tracking for a player"""
        if player_id not in self.player_positions:
            self.player_positions[player_id] = []
            self.player_heatmaps[player_id] = np.zeros(
                (self.heatmap_grid_size, self.heatmap_grid_size)
            )

        self.player_positions[player_id].append(MovementFrame(frame=frame, position=position))

        # Update heatmap
        self._update_heatmap(player_id, position)

    def _update_heatmap(self, player_id: int, position: tuple[float, float]):
        """Update heatmap grid for player position"""
        x, y = position
        grid_x = int(min(x / self.court_width * self.heatmap_grid_size, 
                        self.heatmap_grid_size - 1))
        grid_y = int(min(y / self.court_length * self.heatmap_grid_size, 
                        self.heatmap_grid_size - 1))
        grid_x = max(0, grid_x)
        grid_y = max(0, grid_y)
        self.player_heatmaps[player_id][grid_y, grid_x] += 1

    def _calculate_total_distance(self, positions: list[MovementFrame]) -> float:
        """Calculate total distance covered in meters, filtering outlier jumps"""
        if len(positions) < 2:
            return 0.0
        
        # Max realistic speed for padel: ~8 m/s (sprint), allow some margin
        max_speed_ms = 10.0
        
        total = 0.0
        for i in range(1, len(positions)):
            dx = positions[i].position[0] - positions[i-1].position[0]
            dy = positions[i].position[1] - positions[i-1].position[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # Filter unrealistic jumps
            frame_diff = positions[i].frame - positions[i-1].frame
            if frame_diff > 0:
                speed = dist / (frame_diff / self.fps)
                if speed > max_speed_ms:
                    continue  # Skip this segment - likely tracking error
            
            total += dist
        return total

    def _calculate_horizontal_coverage(self, positions: list[MovementFrame]) -> float:
        """Calculate horizontal (side-to-side) coverage as percentage of court width"""
        if not positions:
            return 0.0
        x_positions = [p.position[0] for p in positions]
        x_range = max(x_positions) - min(x_positions)
        return min(100.0, (x_range / self.court_width) * 100)

    def _calculate_vertical_coverage(self, positions: list[MovementFrame]) -> float:
        """Calculate vertical (front-to-back) coverage as percentage of half court"""
        if not positions:
            return 0.0
        y_positions = [p.position[1] for p in positions]
        y_range = max(y_positions) - min(y_positions)
        half_court = self.court_length / 2
        return min(100.0, (y_range / half_court) * 100)

    def _calculate_avg_speed(self, positions: list[MovementFrame]) -> float:
        """Calculate average movement speed in m/s"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = self._calculate_total_distance(positions)
        total_frames = positions[-1].frame - positions[0].frame
        if total_frames == 0:
            return 0.0
        
        total_time = total_frames / self.fps
        return total_distance / total_time

    def _calculate_max_speed(self, positions: list[MovementFrame]) -> float:
        """Calculate maximum instantaneous speed in m/s (capped at realistic max)"""
        if len(positions) < 2:
            return 0.0
        
        # Cap at max realistic padel sprint speed
        max_realistic_speed = 8.0  # m/s
        
        max_speed = 0.0
        for i in range(1, len(positions)):
            dx = positions[i].position[0] - positions[i-1].position[0]
            dy = positions[i].position[1] - positions[i-1].position[1]
            dist = math.sqrt(dx**2 + dy**2)
            frame_diff = positions[i].frame - positions[i-1].frame
            if frame_diff > 0:
                speed = dist / (frame_diff / self.fps)
                if speed <= max_realistic_speed:
                    max_speed = max(max_speed, speed)
        return max_speed

    def _calculate_sprints(self, positions: list[MovementFrame], sprint_threshold: float = 4.0) -> int:
        """Count number of sprints (movements > threshold m/s)"""
        if len(positions) < 2:
            return 0
        
        sprints = 0
        in_sprint = False
        for i in range(1, len(positions)):
            dx = positions[i].position[0] - positions[i-1].position[0]
            dy = positions[i].position[1] - positions[i-1].position[1]
            dist = math.sqrt(dx**2 + dy**2)
            frame_diff = positions[i].frame - positions[i-1].frame
            if frame_diff > 0:
                speed = dist / (frame_diff / self.fps)
                if speed > sprint_threshold and not in_sprint:
                    sprints += 1
                    in_sprint = True
                elif speed <= sprint_threshold:
                    in_sprint = False
        return sprints

    def _get_team_formation(self, team_positions: dict[int, list[MovementFrame]]) -> dict:
        """
        Analyze team formation (doubles positioning).
        Determines if team plays side-by-side, front-back, or mixed.
        """
        player_ids = list(team_positions.keys())
        if len(player_ids) < 2:
            return {"formation": "unknown", "horizontal_gap": 0, "vertical_gap": 0}

        p1_positions = team_positions[player_ids[0]]
        p2_positions = team_positions[player_ids[1]]

        if not p1_positions or not p2_positions:
            return {"formation": "unknown", "horizontal_gap": 0, "vertical_gap": 0}

        # Use average positions
        p1_avg_x = np.mean([p.position[0] for p in p1_positions])
        p1_avg_y = np.mean([p.position[1] for p in p1_positions])
        p2_avg_x = np.mean([p.position[0] for p in p2_positions])
        p2_avg_y = np.mean([p.position[1] for p in p2_positions])

        h_gap = abs(p1_avg_x - p2_avg_x)
        v_gap = abs(p1_avg_y - p2_avg_y)

        # Classify formation
        if h_gap > v_gap * 1.5:
            formation = "side_by_side"
        elif v_gap > h_gap * 1.5:
            formation = "front_back"
        else:
            formation = "diagonal"

        return {
            "formation": formation,
            "horizontal_gap_m": round(h_gap, 2),
            "vertical_gap_m": round(v_gap, 2),
            "avg_distance_between_partners_m": round(math.sqrt(h_gap**2 + v_gap**2), 2),
        }

    def get_movement_stats(self) -> dict:
        """Get comprehensive movement statistics"""
        stats = {"players": {}, "team_stats": {}}

        for player_id, positions in self.player_positions.items():
            total_distance = self._calculate_total_distance(positions)
            h_coverage = self._calculate_horizontal_coverage(positions)
            v_coverage = self._calculate_vertical_coverage(positions)
            avg_speed = self._calculate_avg_speed(positions)
            max_speed = self._calculate_max_speed(positions)
            sprints = self._calculate_sprints(positions)

            # Average position (center of gravity)
            avg_x = np.mean([p.position[0] for p in positions]) if positions else 0
            avg_y = np.mean([p.position[1] for p in positions]) if positions else 0

            stats["players"][player_id] = {
                "total_distance_m": round(total_distance, 2),
                "horizontal_coverage_pct": round(h_coverage, 1),
                "vertical_coverage_pct": round(v_coverage, 1),
                "avg_speed_ms": round(avg_speed, 2),
                "avg_speed_kmh": round(avg_speed * 3.6, 2),
                "max_speed_ms": round(max_speed, 2),
                "max_speed_kmh": round(max_speed * 3.6, 2),
                "sprints": sprints,
                "avg_position": (round(avg_x, 2), round(avg_y, 2)),
                "heatmap": self.player_heatmaps.get(player_id, np.zeros((10, 10))).tolist(),
            }

        # Team formation analysis
        team_1_positions = {
            pid: pos for pid, pos in self.player_positions.items() 
            if pid in (1, 2)
        }
        team_2_positions = {
            pid: pos for pid, pos in self.player_positions.items() 
            if pid in (3, 4)
        }

        stats["team_stats"]["team_1"] = {
            "formation": self._get_team_formation(team_1_positions),
            "combined_horizontal_coverage": self._team_horizontal_coverage(team_1_positions),
            "combined_vertical_coverage": self._team_vertical_coverage(team_1_positions),
            "total_team_distance_m": round(
                sum(self._calculate_total_distance(pos) for pos in team_1_positions.values()), 2
            ),
        }
        stats["team_stats"]["team_2"] = {
            "formation": self._get_team_formation(team_2_positions),
            "combined_horizontal_coverage": self._team_horizontal_coverage(team_2_positions),
            "combined_vertical_coverage": self._team_vertical_coverage(team_2_positions),
            "total_team_distance_m": round(
                sum(self._calculate_total_distance(pos) for pos in team_2_positions.values()), 2
            ),
        }

        return stats

    def _team_horizontal_coverage(self, team_positions: dict[int, list[MovementFrame]]) -> float:
        """Combined horizontal coverage for a team"""
        all_x = []
        for positions in team_positions.values():
            all_x.extend([p.position[0] for p in positions])
        if not all_x:
            return 0.0
        x_range = max(all_x) - min(all_x)
        return round(min(100.0, (x_range / self.court_width) * 100), 1)

    def _team_vertical_coverage(self, team_positions: dict[int, list[MovementFrame]]) -> float:
        """Combined vertical coverage for a team"""
        all_y = []
        for positions in team_positions.values():
            all_y.extend([p.position[1] for p in positions])
        if not all_y:
            return 0.0
        y_range = max(all_y) - min(all_y)
        half_court = self.court_length / 2
        return round(min(100.0, (y_range / half_court) * 100), 1)
