"""
Video Highlights Module
Automatically identifies and extracts highlight moments from match video.
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import cv2
from pathlib import Path


@dataclass
class HighlightMoment:
    """A notable moment in the match worth highlighting"""
    start_frame: int
    end_frame: int
    category: str  # "winner", "rally", "smash", "save", "ace", "break_point"
    description: str
    importance_score: float  # 0-1 score for highlight priority
    player_id: Optional[int] = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames / fps if fps > 0 else 0.0

    def serialize(self, fps: float) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": round(self.start_frame / fps, 2) if fps > 0 else 0,
            "end_time_s": round(self.end_frame / fps, 2) if fps > 0 else 0,
            "duration_s": round(self.duration_seconds(fps), 2),
            "category": self.category,
            "description": self.description,
            "importance_score": round(self.importance_score, 3),
            "player_id": self.player_id,
        }


class HighlightGenerator:
    """
    Identifies highlight-worthy moments based on:
    - High-speed shots (smashes, fast serves)
    - Long rallies
    - Spectacular saves (defensive plays)
    - Winners (unreturnable shots)
    - Aces
    - Critical game moments
    """

    # Highlight detection thresholds
    LONG_RALLY_THRESHOLD = 8  # hits per rally to count as "long"
    HIGH_SPEED_THRESHOLD = 70.0  # km/h for speed highlights
    HIGHLIGHT_BUFFER_FRAMES = 30  # extra frames before/after moment
    MAX_HIGHLIGHTS = 20  # maximum highlights to generate
    MIN_HIGHLIGHT_DURATION = 30  # minimum frames for a highlight

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.highlights: list[HighlightMoment] = []

    def reset(self):
        self.highlights = []

    def add_winner(self, frame: int, player_id: int, shot_type: str, speed_kmh: float):
        """Add a winner highlight"""
        importance = min(1.0, 0.7 + (speed_kmh / 200.0))
        self.highlights.append(HighlightMoment(
            start_frame=max(0, frame - self.HIGHLIGHT_BUFFER_FRAMES * 3),
            end_frame=frame + self.HIGHLIGHT_BUFFER_FRAMES,
            category="winner",
            description=f"Player {player_id} hits a {shot_type} winner at {speed_kmh:.0f} km/h",
            importance_score=importance,
            player_id=player_id,
        ))

    def add_fast_shot(self, frame: int, player_id: int, shot_type: str, speed_kmh: float):
        """Add a high-speed shot highlight"""
        if speed_kmh >= self.HIGH_SPEED_THRESHOLD:
            importance = min(1.0, 0.5 + (speed_kmh / 200.0))
            self.highlights.append(HighlightMoment(
                start_frame=max(0, frame - self.HIGHLIGHT_BUFFER_FRAMES * 2),
                end_frame=frame + self.HIGHLIGHT_BUFFER_FRAMES,
                category="fast_shot",
                description=f"Player {player_id} fires a {speed_kmh:.0f} km/h {shot_type}",
                importance_score=importance,
                player_id=player_id,
            ))

    def add_smash(self, frame: int, player_id: int, speed_kmh: float):
        """Add smash highlight"""
        importance = min(1.0, 0.6 + (speed_kmh / 180.0))
        self.highlights.append(HighlightMoment(
            start_frame=max(0, frame - self.HIGHLIGHT_BUFFER_FRAMES * 2),
            end_frame=frame + self.HIGHLIGHT_BUFFER_FRAMES,
            category="smash",
            description=f"Player {player_id} smashes at {speed_kmh:.0f} km/h",
            importance_score=importance,
            player_id=player_id,
        ))

    def add_long_rally(self, start_frame: int, end_frame: int, hits: int, duration_s: float):
        """Add long rally highlight"""
        importance = min(1.0, 0.5 + (hits / 20.0))
        self.highlights.append(HighlightMoment(
            start_frame=max(0, start_frame - self.HIGHLIGHT_BUFFER_FRAMES),
            end_frame=end_frame + self.HIGHLIGHT_BUFFER_FRAMES,
            category="long_rally",
            description=f"Epic {hits}-hit rally lasting {duration_s:.1f} seconds",
            importance_score=importance,
        ))

    def add_defensive_save(self, frame: int, player_id: int, distance_covered: float):
        """Add spectacular defensive save"""
        importance = min(1.0, 0.5 + (distance_covered / 5.0))
        self.highlights.append(HighlightMoment(
            start_frame=max(0, frame - self.HIGHLIGHT_BUFFER_FRAMES * 2),
            end_frame=frame + self.HIGHLIGHT_BUFFER_FRAMES * 2,
            category="defensive_save",
            description=f"Player {player_id} makes an incredible save covering {distance_covered:.1f}m",
            importance_score=importance,
            player_id=player_id,
        ))

    def add_ace(self, frame: int, player_id: int, speed_kmh: float):
        """Add ace/unreturnable serve highlight"""
        importance = min(1.0, 0.7 + (speed_kmh / 200.0))
        self.highlights.append(HighlightMoment(
            start_frame=max(0, frame - self.HIGHLIGHT_BUFFER_FRAMES),
            end_frame=frame + self.HIGHLIGHT_BUFFER_FRAMES * 2,
            category="ace",
            description=f"Player {player_id} serves an ace at {speed_kmh:.0f} km/h",
            importance_score=importance,
            player_id=player_id,
        ))

    def generate_from_stats(self, shots: list, rallies: list):
        """
        Auto-generate highlights from shot and rally data.
        
        Args:
            shots: list of Shot objects from ShotDetector
            rallies: list of Rally objects from RallyAnalyzer
        """
        # Fast shots and winners
        for shot in shots:
            if hasattr(shot, 'speed_kmh') and shot.speed_kmh >= self.HIGH_SPEED_THRESHOLD:
                self.add_fast_shot(
                    shot.frame, shot.player_id, 
                    shot.shot_type.value if hasattr(shot.shot_type, 'value') else str(shot.shot_type),
                    shot.speed_kmh
                )
            if hasattr(shot, 'shot_type'):
                shot_type_val = shot.shot_type.value if hasattr(shot.shot_type, 'value') else str(shot.shot_type)
                if shot_type_val == "smash":
                    self.add_smash(shot.frame, shot.player_id, shot.speed_kmh)
            if hasattr(shot, 'is_winner') and shot.is_winner:
                shot_type_val = shot.shot_type.value if hasattr(shot.shot_type, 'value') else str(shot.shot_type)
                self.add_winner(shot.frame, shot.player_id, shot_type_val, shot.speed_kmh)

        # Long rallies
        for rally in rallies:
            if rally.hits >= self.LONG_RALLY_THRESHOLD and rally.end_frame:
                duration = rally.duration_seconds(self.fps)
                self.add_long_rally(rally.start_frame, rally.end_frame, rally.hits, duration)

    def get_highlights(self, max_count: Optional[int] = None) -> list[dict]:
        """
        Get sorted highlights by importance.
        Merges overlapping highlights.
        """
        if not self.highlights:
            return []

        # Sort by importance
        sorted_highlights = sorted(
            self.highlights, 
            key=lambda h: h.importance_score, 
            reverse=True,
        )

        # Merge overlapping highlights
        merged = self._merge_overlapping(sorted_highlights)

        # Limit count
        limit = max_count or self.MAX_HIGHLIGHTS
        top_highlights = merged[:limit]

        # Sort by time for the output
        top_highlights.sort(key=lambda h: h.start_frame)

        return [h.serialize(self.fps) for h in top_highlights]

    def _merge_overlapping(self, highlights: list[HighlightMoment]) -> list[HighlightMoment]:
        """Merge overlapping highlight segments"""
        if not highlights:
            return []

        # Sort by start frame
        sorted_h = sorted(highlights, key=lambda h: h.start_frame)
        merged = [sorted_h[0]]

        for h in sorted_h[1:]:
            last = merged[-1]
            if h.start_frame <= last.end_frame:
                # Merge: extend end frame, keep higher importance
                merged[-1] = HighlightMoment(
                    start_frame=last.start_frame,
                    end_frame=max(last.end_frame, h.end_frame),
                    category=last.category if last.importance_score >= h.importance_score else h.category,
                    description=last.description if last.importance_score >= h.importance_score else h.description,
                    importance_score=max(last.importance_score, h.importance_score),
                    player_id=last.player_id or h.player_id,
                )
            else:
                merged.append(h)

        return merged

    def export_highlight_clips(
        self, 
        video_path: str, 
        output_dir: str,
        max_clips: int = 10,
    ) -> list[str]:
        """
        Export highlight clips as separate video files.
        
        Returns list of output file paths.
        """
        highlights = self.get_highlights(max_count=max_clips)
        if not highlights:
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        exported_files = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for i, highlight in enumerate(highlights):
            output_file = str(output_path / f"highlight_{i+1}_{highlight['category']}.mp4")
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, highlight["start_frame"])
            for _ in range(highlight["end_frame"] - highlight["start_frame"]):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            exported_files.append(output_file)

        cap.release()
        return exported_files

    def get_match_summary(self) -> dict:
        """Generate a match summary from the de-duplicated highlight moments.

        Uses the same merged set that get_highlights() returns so the summary
        counts match the highlights actually listed (overlapping fast-shot
        windows within a single rally collapse into one moment).
        """
        if not self.highlights:
            return {"summary": "No notable moments detected", "total_highlights": 0}

        merged = self._merge_overlapping(self.highlights)

        categories = {}
        for h in merged:
            categories[h.category] = categories.get(h.category, 0) + 1

        total_highlight_time = sum(h.duration_seconds(self.fps) for h in merged)

        top_moment = max(merged, key=lambda h: h.importance_score)

        return {
            "total_highlights": len(merged),
            "total_highlight_duration_s": round(total_highlight_time, 1),
            "highlight_categories": categories,
            "top_moment": top_moment.serialize(self.fps),
            "summary": self._generate_text_summary(categories),
        }

    def _generate_text_summary(self, categories: dict) -> str:
        """Generate human-readable match summary"""
        parts = []
        if categories.get("winner", 0) > 0:
            parts.append(f"{categories['winner']} winners")
        if categories.get("smash", 0) > 0:
            parts.append(f"{categories['smash']} smashes")
        if categories.get("long_rally", 0) > 0:
            parts.append(f"{categories['long_rally']} long rallies")
        if categories.get("ace", 0) > 0:
            parts.append(f"{categories['ace']} aces")
        if categories.get("defensive_save", 0) > 0:
            parts.append(f"{categories['defensive_save']} spectacular saves")

        if parts:
            return f"Match highlights include: {', '.join(parts)}"
        return "Match analyzed - see detailed stats for insights"
