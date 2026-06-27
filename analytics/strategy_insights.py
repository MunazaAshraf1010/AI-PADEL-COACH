"""
Strategy Insights & AI Recommendations Module
Analyzes winning strategies, strengths/weaknesses, and provides personalized tips.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class StrategyInsight:
    category: str
    insight: str
    confidence: float  # 0-1 confidence in the insight
    priority: str  # "high", "medium", "low"

    def serialize(self) -> dict:
        return {
            "category": self.category,
            "insight": self.insight,
            "confidence": round(self.confidence, 2),
            "priority": self.priority,
        }


@dataclass
class Recommendation:
    area: str
    suggestion: str
    current_performance: str
    target_improvement: str
    priority: str

    def serialize(self) -> dict:
        return {
            "area": self.area,
            "suggestion": self.suggestion,
            "current_performance": self.current_performance,
            "target_improvement": self.target_improvement,
            "priority": self.priority,
        }


class StrategyAnalyzer:
    """
    Generates strategy insights and AI recommendations based on all collected stats.
    
    Provides:
    - Winning strategies identified from match data
    - Player strengths and weaknesses
    - Personalized tips for improvement
    - Shot selection recommendations
    - Movement efficiency suggestions
    - Tactical adjustments
    """

    def __init__(self):
        self.insights: list[StrategyInsight] = []
        self.recommendations: dict[int, list[Recommendation]] = {}

    def reset(self):
        self.insights = []
        self.recommendations = {}

    def analyze(
        self,
        shot_stats: dict,
        movement_stats: dict,
        zone_stats: dict,
        rally_stats: dict,
        pressure_stats: dict,
        player_ratings: dict,
    ) -> dict:
        """
        Run full strategy analysis and generate insights + recommendations.
        """
        self.insights = []
        self.recommendations = {}

        # Analyze each player
        player_ids = set()
        for pid in shot_stats.get("player_stats", {}).keys():
            player_ids.add(pid)
        for pid in movement_stats.get("players", {}).keys():
            player_ids.add(pid)

        for player_id in player_ids:
            self.recommendations[player_id] = []
            self._analyze_shot_selection(player_id, shot_stats)
            self._analyze_movement_efficiency(player_id, movement_stats)
            self._analyze_court_positioning(player_id, zone_stats)
            self._analyze_pressure_game(player_id, pressure_stats)
            self._analyze_rally_performance(player_id, rally_stats)
            self._analyze_strengths_weaknesses(player_id, player_ratings)

        # Team-level insights
        self._analyze_team_strategy(movement_stats, zone_stats)

        return {
            "insights": [i.serialize() for i in self.insights],
            "recommendations": {
                pid: [r.serialize() for r in recs]
                for pid, recs in self.recommendations.items()
            },
            "strengths": self._get_strengths(player_ratings),
            "weaknesses": self._get_weaknesses(player_ratings),
            "winning_strategies": self._identify_winning_strategies(
                shot_stats, movement_stats, pressure_stats
            ),
        }

    def _analyze_shot_selection(self, player_id: int, shot_stats: dict):
        """Analyze shot selection patterns and recommend improvements"""
        player_shots = shot_stats.get("player_stats", {}).get(player_id, {})
        shot_types = player_shots.get("shot_types", {})
        total_shots = player_shots.get("total_shots", 0)

        if total_shots == 0:
            return

        # Check shot variety
        types_used = len(shot_types)
        if types_used < 3:
            self.recommendations[player_id].append(Recommendation(
                area="shot_selection",
                suggestion="Increase shot variety. Try incorporating more lobs and volleys to keep opponents guessing.",
                current_performance=f"Using only {types_used} shot type(s)",
                target_improvement="Use 4-5 different shot types regularly",
                priority="high",
            ))

        # Check if too forehand-heavy
        forehand_pct = shot_types.get("forehand", 0) / total_shots * 100 if total_shots > 0 else 0
        if forehand_pct > 70:
            self.recommendations[player_id].append(Recommendation(
                area="shot_selection",
                suggestion="Develop backhand shots. Overreliance on forehand makes you predictable.",
                current_performance=f"{forehand_pct:.0f}% forehand shots",
                target_improvement="Aim for 40-50% forehand, 25-30% backhand mix",
                priority="high",
            ))

        # Check volley usage
        volley_pct = shot_types.get("volley", 0) / total_shots * 100 if total_shots > 0 else 0
        if volley_pct < 10:
            self.recommendations[player_id].append(Recommendation(
                area="shot_selection",
                suggestion="Approach the net more often. Volleys win points in padel.",
                current_performance=f"Only {volley_pct:.0f}% volley shots",
                target_improvement="Aim for 15-25% of shots at the net",
                priority="medium",
            ))

        # Check smash usage
        smash_pct = shot_types.get("smash", 0) / total_shots * 100 if total_shots > 0 else 0
        if smash_pct > 0:
            self.insights.append(StrategyInsight(
                category="shot_selection",
                insight=f"Player {player_id} uses smashes effectively ({smash_pct:.0f}% of shots)",
                confidence=0.7,
                priority="medium",
            ))

    def _analyze_movement_efficiency(self, player_id: int, movement_stats: dict):
        """Analyze movement patterns and suggest improvements"""
        player_movement = movement_stats.get("players", {}).get(player_id, {})

        h_coverage = player_movement.get("horizontal_coverage_pct", 0)
        v_coverage = player_movement.get("vertical_coverage_pct", 0)
        avg_speed = player_movement.get("avg_speed_kmh", 0)
        sprints = player_movement.get("sprints", 0)

        if h_coverage < 50:
            self.recommendations[player_id].append(Recommendation(
                area="movement",
                suggestion="Improve lateral movement. You're not covering enough court width.",
                current_performance=f"{h_coverage:.0f}% horizontal coverage",
                target_improvement="Aim for 70-90% horizontal court coverage",
                priority="high",
            ))

        if v_coverage < 40:
            self.recommendations[player_id].append(Recommendation(
                area="movement",
                suggestion="Move more between front and back. Transition between zones to create opportunities.",
                current_performance=f"{v_coverage:.0f}% vertical coverage",
                target_improvement="Aim for 60-80% vertical court coverage",
                priority="medium",
            ))

        if avg_speed < 5 and sprints < 10:
            self.recommendations[player_id].append(Recommendation(
                area="movement",
                suggestion="Increase movement intensity. Quick bursts to reach shots improve point-winning.",
                current_performance=f"Avg {avg_speed:.1f} km/h, {sprints} sprints",
                target_improvement="Target 8-12 km/h average with 30+ sprints per match",
                priority="medium",
            ))

    def _analyze_court_positioning(self, player_id: int, zone_stats: dict):
        """Analyze court positioning and zone usage"""
        player_zones = zone_stats.get("players", {}).get(player_id, {})
        zone_pcts = player_zones.get("zone_percentages", {})

        back_pct = zone_pcts.get("back_zone", 0)
        volley_pct = zone_pcts.get("volley_zone", 0)
        transition_pct = zone_pcts.get("transition_zone", 0)

        if back_pct > 70:
            self.recommendations[player_id].append(Recommendation(
                area="positioning",
                suggestion="Move forward more! Spending too much time at the back wall limits offensive options.",
                current_performance=f"{back_pct:.0f}% time in back zone",
                target_improvement="Balance: 30-40% back, 25-35% transition, 25-35% net",
                priority="high",
            ))
            self.insights.append(StrategyInsight(
                category="positioning",
                insight=f"Player {player_id} is overly defensive (back zone: {back_pct:.0f}%)",
                confidence=0.85,
                priority="high",
            ))

        if volley_pct > 50:
            self.insights.append(StrategyInsight(
                category="positioning",
                insight=f"Player {player_id} dominates the net ({volley_pct:.0f}% in volley zone)",
                confidence=0.8,
                priority="medium",
            ))

        if transition_pct > 50:
            self.recommendations[player_id].append(Recommendation(
                area="positioning",
                suggestion="Commit to a position. Too much time in transition zone suggests indecision.",
                current_performance=f"{transition_pct:.0f}% in transition zone",
                target_improvement="Be at net or back - minimize transition time to <25%",
                priority="medium",
            ))

    def _analyze_pressure_game(self, player_id: int, pressure_stats: dict):
        """Analyze pressure application"""
        player_pressure = pressure_stats.get("player_stats", {}).get(player_id, {})

        pressure_rate = player_pressure.get("pressure_rate_applied_pct", 0)
        avg_forced = player_pressure.get("avg_distance_forced_m", 0)

        if pressure_rate < 30:
            self.recommendations[player_id].append(Recommendation(
                area="pressure",
                suggestion="Apply more pressure! Place shots wider and deeper to force opponents to move.",
                current_performance=f"Only {pressure_rate:.0f}% pressure shots",
                target_improvement="Aim for 45-60% of shots forcing >2m opponent movement",
                priority="high",
            ))
        elif pressure_rate > 50:
            self.insights.append(StrategyInsight(
                category="pressure",
                insight=f"Player {player_id} applies excellent pressure ({pressure_rate:.0f}% pressure rate)",
                confidence=0.85,
                priority="medium",
            ))

        # Check if player is being pressured too much
        pressure_received = player_pressure.get("pressure_rate_received_pct", 0)
        if pressure_received > 60:
            self.recommendations[player_id].append(Recommendation(
                area="positioning",
                suggestion="You're being moved too much. Improve anticipation and base position.",
                current_performance=f"{pressure_received:.0f}% of opponent shots force >2m movement",
                target_improvement="Reduce to <40% through better positioning",
                priority="high",
            ))

    def _analyze_rally_performance(self, player_id: int, rally_stats: dict):
        """Analyze rally performance"""
        avg_hits = rally_stats.get("avg_hits_per_rally", 0)
        errors = rally_stats.get("errors", 0)
        total = rally_stats.get("total_rallies", 0)

        if total > 0:
            error_rate = errors / total * 100
            if error_rate > 40:
                self.recommendations[player_id].append(Recommendation(
                    area="consistency",
                    suggestion="Reduce unforced errors. Focus on keeping the ball in play before going for winners.",
                    current_performance=f"{error_rate:.0f}% error rate in rallies",
                    target_improvement="Target <20% error rate",
                    priority="high",
                ))

        if avg_hits < 3:
            self.insights.append(StrategyInsight(
                category="rally",
                insight="Short rallies indicate aggressive play or high error rate",
                confidence=0.7,
                priority="medium",
            ))

    def _analyze_strengths_weaknesses(self, player_id: int, player_ratings: dict):
        """Identify specific strengths and weaknesses from ratings"""
        rating_data = player_ratings.get(player_id, {})
        category_ratings = rating_data.get("category_ratings", {})

        for cat, data in category_ratings.items():
            cat_rating = data.get("rating", 0)
            if cat_rating >= 5.0:
                self.insights.append(StrategyInsight(
                    category="strengths",
                    insight=f"Player {player_id} excels in {cat.replace('_', ' ')} (rating: {cat_rating}/7)",
                    confidence=0.8,
                    priority="low",
                ))
            elif cat_rating < 2.5:
                self.insights.append(StrategyInsight(
                    category="weaknesses",
                    insight=f"Player {player_id} needs work on {cat.replace('_', ' ')} (rating: {cat_rating}/7)",
                    confidence=0.8,
                    priority="high",
                ))

    def _analyze_team_strategy(self, movement_stats: dict, zone_stats: dict):
        """Analyze team-level tactics"""
        team_stats = movement_stats.get("team_stats", {})

        for team_name, team_data in team_stats.items():
            formation = team_data.get("formation", {})
            formation_type = formation.get("formation", "unknown")

            if formation_type == "side_by_side":
                self.insights.append(StrategyInsight(
                    category="team_strategy",
                    insight=f"{team_name} plays side-by-side formation (good defensive coverage)",
                    confidence=0.75,
                    priority="medium",
                ))
            elif formation_type == "front_back":
                self.insights.append(StrategyInsight(
                    category="team_strategy",
                    insight=f"{team_name} plays front-back formation (one net, one back)",
                    confidence=0.75,
                    priority="medium",
                ))

            # Check team spacing
            distance = formation.get("avg_distance_between_partners_m", 0)
            if distance < 2.0:
                self.recommendations.setdefault("team", []).append(Recommendation(
                    area="team_positioning",
                    suggestion=f"{team_name}: Partners too close together. Spread out to cover more court.",
                    current_performance=f"Average {distance:.1f}m between partners",
                    target_improvement="Maintain 3-5m distance between partners",
                    priority="high",
                ))

    def _get_strengths(self, player_ratings: dict) -> dict:
        """Extract top strengths per player"""
        strengths = {}
        for pid, data in player_ratings.items():
            cat_ratings = data.get("category_ratings", {})
            sorted_cats = sorted(cat_ratings.items(), key=lambda x: x[1].get("rating", 0), reverse=True)
            strengths[pid] = [
                {"category": cat.replace("_", " "), "rating": info.get("rating", 0)}
                for cat, info in sorted_cats[:3]
            ]
        return strengths

    def _get_weaknesses(self, player_ratings: dict) -> dict:
        """Extract top weaknesses per player"""
        weaknesses = {}
        for pid, data in player_ratings.items():
            cat_ratings = data.get("category_ratings", {})
            sorted_cats = sorted(cat_ratings.items(), key=lambda x: x[1].get("rating", 0))
            weaknesses[pid] = [
                {"category": cat.replace("_", " "), "rating": info.get("rating", 0)}
                for cat, info in sorted_cats[:3]
            ]
        return weaknesses

    def _identify_winning_strategies(
        self, 
        shot_stats: dict, 
        movement_stats: dict,
        pressure_stats: dict,
    ) -> list[str]:
        """Identify what strategies are working"""
        strategies = []

        # Check if pressure correlates with winning
        overall_pressure = pressure_stats.get("overall_pressure_rate_pct", 0)
        if overall_pressure > 50:
            strategies.append(
                "High-pressure play: forcing opponents to cover >2m per shot is effective"
            )

        # Check net play effectiveness
        avg_speed = shot_stats.get("avg_speed_kmh", 0)
        if avg_speed > 50:
            strategies.append(
                "Aggressive shot speed: fast-paced play is putting opponents under time pressure"
            )

        # Check team formations
        team_stats = movement_stats.get("team_stats", {})
        for team_name, data in team_stats.items():
            formation = data.get("formation", {}).get("formation", "")
            if formation == "front_back":
                strategies.append(
                    f"{team_name}: Front-back formation allows net dominance while maintaining defense"
                )

        if not strategies:
            strategies.append("Build longer rallies to identify opponent patterns")
            strategies.append("Focus on consistent shot placement to create opportunities")

        return strategies
