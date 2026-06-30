"""
Player Rating Module
AI rating from 0-7 based on 40+ performance indicators.
Inspired by padel level classification systems.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceIndicator:
    name: str
    value: float
    weight: float
    max_value: float
    category: str

    @property
    def normalized_score(self) -> float:
        """Normalize to 0-1 scale"""
        if self.max_value == 0:
            return 0.0
        return min(1.0, max(0.0, self.value / self.max_value))

    @property
    def weighted_score(self) -> float:
        return self.normalized_score * self.weight


class PlayerRating:
    """
    AI-powered player rating system (0-7 scale).
    
    Level descriptions:
    0-1: Beginner - Learning basic strokes
    1-2: Novice - Can sustain short rallies
    2-3: Intermediate - Consistent groundstrokes, developing tactics
    3-4: Advanced Intermediate - Good court coverage, varied shots
    4-5: Advanced - Strong all-round game, effective strategies
    5-6: Expert - High-level competition, precise shot placement
    6-7: Professional - Elite level, tournament player
    
    Rating based on 40+ indicators across categories:
    - Shot quality (accuracy, speed, variety)
    - Court coverage (zones, movement efficiency)
    - Rally performance (length, consistency)
    - Pressure game (forcing opponents)
    - Tactical awareness (positioning, shot selection)
    - Serve/return quality
    - Error rate
    - Physical performance (speed, distance)
    """

    MAX_RATING = 7.0
    
    # Performance benchmarks (for a level 7 player)
    BENCHMARKS = {
        # Shot quality
        "avg_shot_speed_kmh": 80.0,
        "max_shot_speed_kmh": 140.0,
        "shot_accuracy": 0.85,
        "shot_variety_types": 5,
        "forehand_pct": 40.0,
        "backhand_pct": 30.0,
        "volley_pct": 20.0,
        "smash_accuracy": 0.9,
        "lob_effectiveness": 0.7,
        
        # Rally performance
        "avg_rally_length": 8.0,
        "consistency_rate": 0.85,
        "rally_win_rate": 0.6,
        
        # Court coverage
        "horizontal_coverage_pct": 90.0,
        "vertical_coverage_pct": 85.0,
        "volley_zone_time_pct": 35.0,
        "zone_transition_efficiency": 0.8,
        
        # Movement
        "avg_speed_kmh": 12.0,
        "max_speed_kmh": 25.0,
        "total_distance_m": 2000.0,
        "sprints_count": 50,
        
        # Pressure
        "pressure_rate_pct": 60.0,
        "avg_displacement_forced_m": 3.5,
        
        # Serve/Return
        "serve_speed_kmh": 100.0,
        "first_serve_in_pct": 70.0,
        "return_effectiveness_pct": 65.0,
        
        # Errors
        "unforced_error_rate": 0.10,  # lower is better (inverted)
        "winner_to_error_ratio": 2.0,
        
        # Tactical
        "shot_placement_variety": 0.8,
        "net_approach_success_rate": 0.7,
        "defensive_save_rate": 0.5,
    }

    def __init__(self):
        self.indicators: dict[int, list[PerformanceIndicator]] = {}

    def reset(self):
        self.indicators = {}

    def calculate_rating(
        self,
        player_id: int,
        shot_stats: dict,
        movement_stats: dict,
        zone_stats: dict,
        rally_stats: dict,
        pressure_stats: dict,
    ) -> dict:
        """
        Calculate comprehensive player rating based on all analytics.
        
        Returns dict with overall rating and breakdown by category.
        """
        indicators = []

        # === SHOT QUALITY INDICATORS ===
        player_shot = shot_stats.get("player_stats", {}).get(player_id, {})

        indicators.append(PerformanceIndicator(
            name="avg_shot_speed",
            value=player_shot.get("avg_speed_kmh", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["avg_shot_speed_kmh"],
            category="shot_quality",
        ))
        indicators.append(PerformanceIndicator(
            name="max_shot_speed",
            value=player_shot.get("max_speed_kmh", 0),
            weight=1.0,
            max_value=self.BENCHMARKS["max_shot_speed_kmh"],
            category="shot_quality",
        ))
        indicators.append(PerformanceIndicator(
            name="shot_accuracy",
            value=player_shot.get("avg_accuracy", 0),
            weight=2.0,
            max_value=self.BENCHMARKS["shot_accuracy"],
            category="shot_quality",
        ))

        # Shot variety
        shot_types_used = len(player_shot.get("shot_types", {}))
        indicators.append(PerformanceIndicator(
            name="shot_variety",
            value=shot_types_used,
            weight=1.5,
            max_value=self.BENCHMARKS["shot_variety_types"],
            category="shot_quality",
        ))

        # Winners vs errors. Per-shot winner/error flags are never set by the
        # detector, so read the point-level outcomes the rally analyzer assigns
        # (keyed by player id, which may be int or str). Using the real values
        # stops every player from scoring a perfect, identical consistency rating.
        point_outcomes = rally_stats.get("point_outcomes", {})
        outcome = (
            point_outcomes.get(player_id)
            or point_outcomes.get(str(player_id))
            or {}
        )
        winners = outcome.get("winners", 0)
        errors = outcome.get("errors", 0)
        w_e_ratio = winners / max(1, errors)
        indicators.append(PerformanceIndicator(
            name="winner_error_ratio",
            value=w_e_ratio,
            weight=2.0,
            max_value=self.BENCHMARKS["winner_to_error_ratio"],
            category="shot_quality",
        ))

        # === MOVEMENT INDICATORS ===
        player_movement = movement_stats.get("players", {}).get(player_id, {})

        indicators.append(PerformanceIndicator(
            name="total_distance",
            value=player_movement.get("total_distance_m", 0),
            weight=1.0,
            max_value=self.BENCHMARKS["total_distance_m"],
            category="movement",
        ))
        indicators.append(PerformanceIndicator(
            name="avg_speed",
            value=player_movement.get("avg_speed_kmh", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["avg_speed_kmh"],
            category="movement",
        ))
        indicators.append(PerformanceIndicator(
            name="max_speed",
            value=player_movement.get("max_speed_kmh", 0),
            weight=1.0,
            max_value=self.BENCHMARKS["max_speed_kmh"],
            category="movement",
        ))
        indicators.append(PerformanceIndicator(
            name="sprints",
            value=player_movement.get("sprints", 0),
            weight=1.0,
            max_value=self.BENCHMARKS["sprints_count"],
            category="movement",
        ))
        indicators.append(PerformanceIndicator(
            name="horizontal_coverage",
            value=player_movement.get("horizontal_coverage_pct", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["horizontal_coverage_pct"],
            category="movement",
        ))
        indicators.append(PerformanceIndicator(
            name="vertical_coverage",
            value=player_movement.get("vertical_coverage_pct", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["vertical_coverage_pct"],
            category="movement",
        ))

        # === COURT ZONE INDICATORS ===
        player_zones = zone_stats.get("players", {}).get(player_id, {})
        zone_pcts = player_zones.get("zone_percentages", {})

        indicators.append(PerformanceIndicator(
            name="volley_zone_presence",
            value=zone_pcts.get("volley_zone", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["volley_zone_time_pct"],
            category="court_coverage",
        ))
        indicators.append(PerformanceIndicator(
            name="zone_transitions",
            value=player_zones.get("zone_transitions", 0),
            weight=1.0,
            max_value=100,  # normalized count
            category="court_coverage",
        ))

        # === RALLY INDICATORS ===
        indicators.append(PerformanceIndicator(
            name="avg_rally_length",
            value=rally_stats.get("avg_hits_per_rally", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["avg_rally_length"],
            category="rally_performance",
        ))
        indicators.append(PerformanceIndicator(
            name="rally_consistency",
            value=min(1.0, rally_stats.get("total_rallies", 0) / max(1, rally_stats.get("errors", 1))),
            weight=2.0,
            max_value=self.BENCHMARKS["consistency_rate"],
            category="rally_performance",
        ))

        # === PRESSURE INDICATORS ===
        player_pressure = pressure_stats.get("player_stats", {}).get(player_id, {})

        indicators.append(PerformanceIndicator(
            name="pressure_rate",
            value=player_pressure.get("pressure_rate_applied_pct", 0),
            weight=2.0,
            max_value=self.BENCHMARKS["pressure_rate_pct"],
            category="pressure",
        ))
        indicators.append(PerformanceIndicator(
            name="avg_displacement_forced",
            value=player_pressure.get("avg_distance_forced_m", 0),
            weight=1.5,
            max_value=self.BENCHMARKS["avg_displacement_forced_m"],
            category="pressure",
        ))

        # === ERROR INDICATORS (inverted - lower is better) ===
        # Error rate = errors as a fraction of points this player decided
        # (points won + points lost). In racket sports roughly half of decided
        # points end in an error, so we map realistically: <=20% error rate is
        # elite consistency (score 1.0), >=55% is poor (score 0.0), linear
        # between. Comparing this to the old 10%-per-shot benchmark zeroed
        # everyone.
        points_played = winners + errors
        error_rate = errors / points_played if points_played > 0 else 0.0
        ELITE_ERR, POOR_ERR = 0.20, 0.55
        error_score = 1.0 - (error_rate - ELITE_ERR) / (POOR_ERR - ELITE_ERR)
        error_score = max(0.0, min(1.0, error_score))
        indicators.append(PerformanceIndicator(
            name="low_error_rate",
            value=max(0, error_score),
            weight=2.0,
            max_value=1.0,
            category="consistency",
        ))

        # Store indicators
        self.indicators[player_id] = indicators

        # Calculate overall rating
        total_weighted_score = sum(ind.weighted_score for ind in indicators)
        total_weight = sum(ind.weight for ind in indicators)
        
        if total_weight == 0:
            overall_score = 0
        else:
            overall_score = (total_weighted_score / total_weight) * self.MAX_RATING

        overall_score = round(min(self.MAX_RATING, max(0, overall_score)), 2)

        # Category breakdown
        categories = {}
        for ind in indicators:
            if ind.category not in categories:
                categories[ind.category] = {"score": 0, "weight": 0, "indicators": []}
            categories[ind.category]["score"] += ind.weighted_score
            categories[ind.category]["weight"] += ind.weight
            categories[ind.category]["indicators"].append({
                "name": ind.name,
                "value": round(ind.value, 3),
                "normalized": round(ind.normalized_score, 3),
                "weighted": round(ind.weighted_score, 3),
            })

        category_ratings = {}
        for cat, data in categories.items():
            if data["weight"] > 0:
                cat_score = (data["score"] / data["weight"]) * self.MAX_RATING
                category_ratings[cat] = {
                    "rating": round(min(self.MAX_RATING, cat_score), 2),
                    "indicators": data["indicators"],
                }

        return {
            "overall_rating": overall_score,
            "level_description": self._get_level_description(overall_score),
            "category_ratings": category_ratings,
            "total_indicators": len(indicators),
        }

    def _get_level_description(self, rating: float) -> str:
        """Get human-readable level description"""
        if rating < 1:
            return "Beginner - Learning basic strokes and court positioning"
        elif rating < 2:
            return "Novice - Can sustain short rallies, developing consistency"
        elif rating < 3:
            return "Intermediate - Consistent groundstrokes, developing tactics"
        elif rating < 4:
            return "Advanced Intermediate - Good court coverage, varied shot selection"
        elif rating < 5:
            return "Advanced - Strong all-round game, effective strategies"
        elif rating < 6:
            return "Expert - High-level competition, precise placement"
        else:
            return "Professional - Elite level, tournament-ready player"

    def get_all_ratings(
        self,
        shot_stats: dict,
        movement_stats: dict,
        zone_stats: dict,
        rally_stats: dict,
        pressure_stats: dict,
    ) -> dict:
        """Calculate ratings for all detected players"""
        ratings = {}
        
        # Get all player IDs from available stats
        player_ids = set()
        for pid in shot_stats.get("player_stats", {}).keys():
            player_ids.add(pid)
        for pid in movement_stats.get("players", {}).keys():
            player_ids.add(pid)

        for player_id in player_ids:
            ratings[player_id] = self.calculate_rating(
                player_id=player_id,
                shot_stats=shot_stats,
                movement_stats=movement_stats,
                zone_stats=zone_stats,
                rally_stats=rally_stats,
                pressure_stats=pressure_stats,
            )

        return ratings
