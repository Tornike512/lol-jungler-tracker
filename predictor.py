"""
Jungle pathing prediction logic.
Predicts enemy jungler location based on last known position and typical jungle patterns.
"""
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from zones import Zone, get_zone_by_name, get_adjacent_zones, ThreatLevel


class JungleSide(Enum):
    """Which side of the jungle."""
    BLUE = "blue"  # Top side (Blue buff, Gromp, Wolves)
    RED = "red"    # Bot side (Red buff, Krugs, Raptors)
    UNKNOWN = "unknown"


@dataclass
class Prediction:
    """A prediction of jungler location."""
    zone: Zone
    confidence: float  # 0.0 to 1.0
    reasoning: str
    time_since_seen: float


# Standard jungle clear times (seconds from one camp to the next)
CAMP_CLEAR_TIMES = {
    # Blue side clear times
    ("enemy_blue", "enemy_gromp"): 8,
    ("enemy_gromp", "enemy_blue"): 8,
    ("enemy_blue", "enemy_wolves"): 12,
    ("enemy_wolves", "enemy_blue"): 12,
    ("enemy_gromp", "enemy_wolves"): 10,
    ("enemy_wolves", "enemy_gromp"): 10,

    # Red side clear times
    ("enemy_red", "enemy_krugs"): 10,
    ("enemy_krugs", "enemy_red"): 10,
    ("enemy_red", "enemy_raptors"): 8,
    ("enemy_raptors", "enemy_red"): 8,
    ("enemy_krugs", "enemy_raptors"): 12,
    ("enemy_raptors", "enemy_krugs"): 12,

    # Cross-jungle times
    ("enemy_wolves", "enemy_raptors"): 15,
    ("enemy_raptors", "enemy_wolves"): 15,
    ("enemy_blue", "enemy_red"): 25,
    ("enemy_red", "enemy_blue"): 25,

    # River traversal
    ("enemy_blue", "baron_pit"): 10,
    ("enemy_red", "dragon_pit"): 10,
    ("enemy_wolves", "top_river"): 8,
    ("enemy_raptors", "bot_river"): 8,

    # Gank timings
    ("enemy_wolves", "mid_river_top"): 6,
    ("enemy_raptors", "mid_river_bot"): 6,
    ("enemy_blue", "top_lane_river"): 12,
    ("enemy_red", "bot_lane_river"): 12,
}

# Common jungle paths from starting positions
STANDARD_PATHS = {
    "enemy_blue": [
        ["enemy_gromp", "enemy_wolves", "enemy_raptors", "enemy_red", "enemy_krugs"],
        ["enemy_gromp", "enemy_wolves", "mid_river_top"],  # Level 3 gank
        ["enemy_wolves", "enemy_raptors", "enemy_red"],
    ],
    "enemy_red": [
        ["enemy_krugs", "enemy_raptors", "enemy_wolves", "enemy_blue", "enemy_gromp"],
        ["enemy_raptors", "enemy_wolves", "mid_river_top"],  # Level 3 gank
        ["enemy_krugs", "enemy_raptors", "bot_river"],
    ],
    "enemy_raptors": [
        ["enemy_red", "enemy_krugs"],
        ["enemy_wolves", "enemy_blue"],
        ["mid_river_bot", "bot_river"],
        ["mid_river_top", "top_river"],
    ],
    "enemy_wolves": [
        ["enemy_blue", "enemy_gromp"],
        ["enemy_raptors", "enemy_red"],
        ["mid_river_top", "top_river"],
    ],
}

# Zones that suggest ganking intent
GANK_INDICATOR_ZONES = {
    "top_river": ["top_tribush", "top_lane_river"],
    "bot_river": ["bot_tribush", "bot_lane_river"],
    "mid_river_top": ["mid_center", "top_pixel_brush"],
    "mid_river_bot": ["mid_center", "bot_pixel_brush"],
    "enemy_raptors": ["mid_river_bot", "mid_outer_enemy"],
    "enemy_wolves": ["mid_river_top", "mid_outer_enemy"],
}


class JunglePredictor:
    """Predicts enemy jungler location based on pathing logic."""

    def __init__(self):
        self._last_known_zone: Optional[str] = None
        self._last_known_time: float = 0
        self._path_history: List[str] = []
        self._max_history = 10

    def update_position(self, zone_name: str, timestamp: float):
        """Update with a confirmed jungler position."""
        if zone_name != self._last_known_zone:
            self._path_history.append(zone_name)
            if len(self._path_history) > self._max_history:
                self._path_history.pop(0)

        self._last_known_zone = zone_name
        self._last_known_time = timestamp

    def predict(self, current_time: float) -> Optional[Prediction]:
        """
        Predict current jungler location based on last known position and time elapsed.

        Returns:
            Prediction object with likely zone and confidence
        """
        if not self._last_known_zone:
            return None

        time_elapsed = current_time - self._last_known_time
        last_zone = get_zone_by_name(self._last_known_zone)

        if last_zone is None:
            return None

        # Very recent - they're probably still in the same area
        if time_elapsed < 5:
            return Prediction(
                zone=last_zone,
                confidence=0.9,
                reasoning="Just seen here",
                time_since_seen=time_elapsed
            )

        # Check if they were at a jungle camp
        if last_zone.is_jungle_camp:
            return self._predict_from_camp(last_zone, time_elapsed)

        # Check if they were in river (likely ganking or rotating)
        if "river" in self._last_known_zone:
            return self._predict_from_river(last_zone, time_elapsed)

        # Generic prediction based on adjacent zones
        return self._predict_generic(last_zone, time_elapsed)

    def _predict_from_camp(self, camp_zone: Zone, time_elapsed: float) -> Prediction:
        """Predict location when last seen at a jungle camp."""
        camp_name = camp_zone.name

        # Get possible next locations from standard paths
        possible_paths = STANDARD_PATHS.get(camp_name, [])
        possible_next = set()

        for path in possible_paths:
            if path:
                possible_next.add(path[0])

        # Consider clear times to narrow down
        best_guess = None
        best_confidence = 0.0

        for next_zone_name in possible_next:
            key = (camp_name, next_zone_name)
            travel_time = CAMP_CLEAR_TIMES.get(key, 15)

            # If elapsed time roughly matches travel time, high confidence
            time_diff = abs(time_elapsed - travel_time)

            if time_diff < 5:
                confidence = 0.7 - (time_diff * 0.05)
            elif time_diff < 10:
                confidence = 0.5 - (time_diff * 0.03)
            else:
                confidence = 0.3

            if confidence > best_confidence:
                next_zone = get_zone_by_name(next_zone_name)
                if next_zone:
                    best_guess = next_zone
                    best_confidence = confidence

        if best_guess:
            return Prediction(
                zone=best_guess,
                confidence=best_confidence,
                reasoning=f"Likely cleared {camp_zone.display_name}, moving to next camp",
                time_since_seen=time_elapsed
            )

        # Fallback: they're somewhere in the jungle
        # After 30+ seconds, could be anywhere
        if time_elapsed > 30:
            return Prediction(
                zone=camp_zone,
                confidence=0.2,
                reasoning="Unknown location - been too long",
                time_since_seen=time_elapsed
            )

        return Prediction(
            zone=camp_zone,
            confidence=0.4,
            reasoning=f"Possibly still near {camp_zone.display_name}",
            time_since_seen=time_elapsed
        )

    def _predict_from_river(self, river_zone: Zone, time_elapsed: float) -> Prediction:
        """Predict location when last seen in river (likely ganking)."""
        zone_name = river_zone.name

        # If they were in river, they're likely:
        # 1. Ganking a lane
        # 2. Taking an objective
        # 3. Returning to jungle

        gank_targets = GANK_INDICATOR_ZONES.get(zone_name, [])

        if time_elapsed < 8:
            # Probably committing to the gank or backing off
            confidence = 0.6 - (time_elapsed * 0.05)
            return Prediction(
                zone=river_zone,
                confidence=max(0.3, confidence),
                reasoning="May be ganking or backing off",
                time_since_seen=time_elapsed
            )

        # After 8+ seconds, they've either committed or returned to jungle
        if gank_targets:
            target_zone = get_zone_by_name(gank_targets[0])
            if target_zone:
                return Prediction(
                    zone=target_zone,
                    confidence=0.4,
                    reasoning="May have ganked and is clearing or backing",
                    time_since_seen=time_elapsed
                )

        return Prediction(
            zone=river_zone,
            confidence=0.25,
            reasoning="Lost track - could be anywhere",
            time_since_seen=time_elapsed
        )

    def _predict_generic(self, last_zone: Zone, time_elapsed: float) -> Prediction:
        """Generic prediction based on zone adjacency and time."""
        # Get adjacent zones
        adjacent = get_adjacent_zones(last_zone.name)

        if adjacent and time_elapsed < 15:
            # Probably moved to adjacent zone
            # Prioritize zones with higher threat to mid
            best_adjacent = max(adjacent, key=lambda z: z.threat_level.value)
            confidence = 0.5 - (time_elapsed * 0.02)
            return Prediction(
                zone=best_adjacent,
                confidence=max(0.2, confidence),
                reasoning=f"Likely moved from {last_zone.display_name}",
                time_since_seen=time_elapsed
            )

        # Fallback
        return Prediction(
            zone=last_zone,
            confidence=max(0.1, 0.4 - (time_elapsed * 0.01)),
            reasoning="Uncertain - last seen here",
            time_since_seen=time_elapsed
        )

    def get_threat_assessment(self, prediction: Prediction) -> Tuple[str, ThreatLevel]:
        """
        Assess the threat level of a prediction for mid lane.

        Returns:
            Tuple of (threat_message, threat_level)
        """
        zone = prediction.zone

        if zone.threat_level == ThreatLevel.DANGER:
            return (f"DANGER - {zone.display_name}", ThreatLevel.DANGER)
        elif zone.threat_level == ThreatLevel.HIGH:
            return (f"Caution - {zone.display_name}", ThreatLevel.HIGH)
        elif zone.threat_level == ThreatLevel.MEDIUM:
            return (f"Nearby - {zone.display_name}", ThreatLevel.MEDIUM)
        else:
            return (f"Safe - {zone.display_name}", ThreatLevel.LOW)

    def reset(self):
        """Reset predictor state for new game."""
        self._last_known_zone = None
        self._last_known_time = 0
        self._path_history = []


# Global predictor instance
jungle_predictor = JunglePredictor()
