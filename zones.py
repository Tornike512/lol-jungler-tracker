"""
Map zone definitions and position mapping for League of Legends minimap.

The minimap is treated as a normalized coordinate system (0.0 to 1.0) where:
- (0, 0) is top-left (enemy base on blue side)
- (1, 1) is bottom-right (allied base on blue side)

Zone boundaries are defined as polygons or rectangles in this coordinate space.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class ThreatLevel(Enum):
    """Threat level for mid lane player."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    DANGER = 4


@dataclass
class Zone:
    """Represents a map zone with boundaries and metadata."""
    name: str
    display_name: str
    # Bounding box: (x_min, y_min, x_max, y_max) in normalized coords
    bounds: Tuple[float, float, float, float]
    threat_level: ThreatLevel = ThreatLevel.LOW
    is_jungle_camp: bool = False
    adjacent_zones: List[str] = None

    def __post_init__(self):
        if self.adjacent_zones is None:
            self.adjacent_zones = []

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this zone."""
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max


# Define all map zones with normalized coordinates
# Coordinates are approximate and based on standard minimap layout
ZONES: List[Zone] = [
    # === BASES ===
    Zone("enemy_base", "enemy base", (0.0, 0.0, 0.15, 0.15), ThreatLevel.NONE),
    Zone("allied_base", "allied base", (0.85, 0.85, 1.0, 1.0), ThreatLevel.NONE),
    Zone("enemy_fountain", "enemy fountain", (0.0, 0.0, 0.08, 0.08), ThreatLevel.NONE),
    Zone("allied_fountain", "allied fountain", (0.92, 0.92, 1.0, 1.0), ThreatLevel.NONE),

    # === TOP LANE ===
    Zone("top_outer", "top outer turret", (0.08, 0.20, 0.18, 0.30), ThreatLevel.LOW),
    Zone("top_inner", "top inner turret", (0.15, 0.12, 0.25, 0.22), ThreatLevel.LOW),
    Zone("top_inhib", "top inhibitor", (0.05, 0.05, 0.15, 0.15), ThreatLevel.NONE),
    Zone("top_lane_river", "top lane near river", (0.18, 0.28, 0.28, 0.38), ThreatLevel.MEDIUM),

    # === MID LANE ===
    Zone("mid_outer_enemy", "enemy mid outer", (0.30, 0.30, 0.42, 0.42), ThreatLevel.MEDIUM),
    Zone("mid_inner_enemy", "enemy mid inner", (0.22, 0.22, 0.32, 0.32), ThreatLevel.LOW),
    Zone("mid_center", "mid lane center", (0.42, 0.42, 0.58, 0.58), ThreatLevel.HIGH),
    Zone("mid_outer_allied", "allied mid outer", (0.58, 0.58, 0.70, 0.70), ThreatLevel.MEDIUM),
    Zone("mid_inner_allied", "allied mid inner", (0.68, 0.68, 0.78, 0.78), ThreatLevel.LOW),

    # === BOT LANE ===
    Zone("bot_outer", "bot outer turret", (0.70, 0.82, 0.80, 0.92), ThreatLevel.LOW),
    Zone("bot_inner", "bot inner turret", (0.78, 0.75, 0.88, 0.85), ThreatLevel.LOW),
    Zone("bot_inhib", "bot inhibitor", (0.85, 0.85, 0.95, 0.95), ThreatLevel.NONE),
    Zone("bot_lane_river", "bot lane near river", (0.62, 0.72, 0.72, 0.82), ThreatLevel.MEDIUM),

    # === RIVER ===
    Zone("top_river", "top river", (0.25, 0.35, 0.40, 0.50), ThreatLevel.HIGH),
    Zone("bot_river", "bot river", (0.50, 0.60, 0.65, 0.75), ThreatLevel.HIGH),
    Zone("mid_river_top", "mid river top side", (0.38, 0.45, 0.48, 0.55), ThreatLevel.DANGER,
         adjacent_zones=["top_river", "mid_center"]),
    Zone("mid_river_bot", "mid river bot side", (0.45, 0.52, 0.55, 0.62), ThreatLevel.DANGER,
         adjacent_zones=["bot_river", "mid_center"]),

    # === RIVER OBJECTIVES ===
    Zone("baron_pit", "Baron pit", (0.22, 0.38, 0.32, 0.48), ThreatLevel.MEDIUM,
         adjacent_zones=["top_river"]),
    Zone("dragon_pit", "Dragon pit", (0.52, 0.68, 0.62, 0.78), ThreatLevel.MEDIUM,
         adjacent_zones=["bot_river"]),

    # === BRUSHES ===
    Zone("top_tribush", "top tri-brush", (0.20, 0.32, 0.28, 0.40), ThreatLevel.HIGH),
    Zone("bot_tribush", "bot tri-brush", (0.60, 0.72, 0.68, 0.80), ThreatLevel.HIGH),
    Zone("top_pixel_brush", "top pixel brush", (0.35, 0.42, 0.42, 0.50), ThreatLevel.DANGER,
         adjacent_zones=["mid_river_top", "top_river"]),
    Zone("bot_pixel_brush", "bot pixel brush", (0.50, 0.58, 0.58, 0.65), ThreatLevel.DANGER,
         adjacent_zones=["mid_river_bot", "bot_river"]),

    # === ENEMY JUNGLE (Top side - Blue buff side) ===
    Zone("enemy_blue", "enemy blue buff", (0.18, 0.22, 0.28, 0.32), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("enemy_gromp", "enemy gromp", (0.12, 0.28, 0.20, 0.36), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("enemy_wolves", "enemy wolves", (0.22, 0.30, 0.32, 0.40), ThreatLevel.DANGER,
         is_jungle_camp=True, adjacent_zones=["mid_outer_enemy"]),

    # === ENEMY JUNGLE (Bot side - Red buff side) ===
    Zone("enemy_red", "enemy red buff", (0.32, 0.12, 0.42, 0.22), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("enemy_krugs", "enemy krugs", (0.38, 0.05, 0.48, 0.15), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("enemy_raptors", "enemy raptors", (0.28, 0.20, 0.38, 0.30), ThreatLevel.DANGER,
         is_jungle_camp=True, adjacent_zones=["mid_outer_enemy"]),

    # === ALLIED JUNGLE (Top side - Red buff side) ===
    Zone("allied_red", "allied red buff", (0.58, 0.78, 0.68, 0.88), ThreatLevel.MEDIUM,
         is_jungle_camp=True),
    Zone("allied_krugs", "allied krugs", (0.52, 0.85, 0.62, 0.95), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("allied_raptors", "allied raptors", (0.62, 0.70, 0.72, 0.80), ThreatLevel.DANGER,
         is_jungle_camp=True, adjacent_zones=["mid_outer_allied"]),

    # === ALLIED JUNGLE (Bot side - Blue buff side) ===
    Zone("allied_blue", "allied blue buff", (0.72, 0.68, 0.82, 0.78), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("allied_gromp", "allied gromp", (0.80, 0.64, 0.88, 0.72), ThreatLevel.LOW,
         is_jungle_camp=True),
    Zone("allied_wolves", "allied wolves", (0.68, 0.60, 0.78, 0.70), ThreatLevel.DANGER,
         is_jungle_camp=True, adjacent_zones=["mid_outer_allied"]),
]

# Create lookup dictionary
ZONE_MAP = {zone.name: zone for zone in ZONES}


def get_zone_at_position(x: float, y: float) -> Optional[Zone]:
    """
    Get the zone at a given normalized position.
    Returns the most specific zone (smallest area) if multiple zones overlap.
    """
    matching_zones = []
    for zone in ZONES:
        if zone.contains(x, y):
            # Calculate zone area for specificity
            x_min, y_min, x_max, y_max = zone.bounds
            area = (x_max - x_min) * (y_max - y_min)
            matching_zones.append((zone, area))

    if not matching_zones:
        return None

    # Return smallest (most specific) zone
    matching_zones.sort(key=lambda z: z[1])
    return matching_zones[0][0]


def get_zone_by_name(name: str) -> Optional[Zone]:
    """Get a zone by its internal name."""
    return ZONE_MAP.get(name)


def is_danger_zone(zone_name: str) -> bool:
    """Check if a zone is considered dangerous for mid lane."""
    zone = ZONE_MAP.get(zone_name)
    return zone is not None and zone.threat_level == ThreatLevel.DANGER


def get_adjacent_zones(zone_name: str) -> List[Zone]:
    """Get zones adjacent to the given zone."""
    zone = ZONE_MAP.get(zone_name)
    if zone is None or not zone.adjacent_zones:
        return []
    return [ZONE_MAP[name] for name in zone.adjacent_zones if name in ZONE_MAP]


def get_jungle_camps() -> List[Zone]:
    """Get all jungle camp zones."""
    return [zone for zone in ZONES if zone.is_jungle_camp]


def normalize_position(x: int, y: int, minimap_width: int, minimap_height: int) -> Tuple[float, float]:
    """
    Convert pixel position on minimap to normalized coordinates.

    Args:
        x: Pixel x position on minimap
        y: Pixel y position on minimap
        minimap_width: Width of minimap in pixels
        minimap_height: Height of minimap in pixels

    Returns:
        Tuple of (normalized_x, normalized_y) in range [0, 1]
    """
    norm_x = x / minimap_width if minimap_width > 0 else 0
    norm_y = y / minimap_height if minimap_height > 0 else 0
    return (min(1.0, max(0.0, norm_x)), min(1.0, max(0.0, norm_y)))
