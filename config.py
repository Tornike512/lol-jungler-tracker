"""
Configuration settings and calibration data for the LoL Jungler Tracker.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CHAMPIONS_DIR = os.path.join(ASSETS_DIR, "champions")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_FILE = os.path.join(BASE_DIR, "calibration.json")

# Ensure directories exist
os.makedirs(CHAMPIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


@dataclass
class MinimapCalibration:
    """Stores minimap location and size data."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    side: str = "right"  # "left" or "right"

    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, width, height) tuple for screen capture."""
        return (self.x, self.y, self.width, self.height)

    @property
    def is_valid(self) -> bool:
        """Check if calibration data is valid."""
        return self.width > 0 and self.height > 0


@dataclass
class TrackerSettings:
    """Main tracker configuration settings."""
    # Detection settings
    scan_rate_hz: int = 15  # Scans per second
    template_threshold: float = 0.7  # Match confidence threshold

    # Alert settings
    alert_cooldown_seconds: float = 10.0
    prediction_delay_seconds: float = 15.0  # Time before switching to prediction mode

    # Mid lane priority - zones that get DANGER prefix
    mid_danger_zones: list = field(default_factory=lambda: [
        "enemy_raptors", "allied_raptors",
        "mid_river_top", "mid_river_bot",
        "top_pixel_brush", "bot_pixel_brush",
        "enemy_wolves", "allied_wolves"
    ])

    # Voice settings
    voice_enabled: bool = True
    voice_rate: int = 175  # Words per minute

    # Overlay settings
    overlay_enabled: bool = True
    overlay_position: Tuple[int, int] = (10, 10)
    overlay_opacity: float = 0.8

    # Game detection
    game_window_title: str = "League of Legends (TM) Client"
    game_process_name: str = "League of Legends.exe"


@dataclass
class TrackerState:
    """Runtime state of the tracker."""
    is_running: bool = False
    is_game_active: bool = False
    enemy_jungler: str = ""
    last_seen_position: Optional[Tuple[float, float]] = None
    last_seen_time: Optional[float] = None
    last_seen_zone: str = ""
    last_alert_time: float = 0.0
    is_jungler_visible: bool = False
    prediction_active: bool = False
    predicted_zone: str = ""
    confidence: str = "none"  # "high", "medium", "low", "none"


class Config:
    """Main configuration manager."""

    def __init__(self):
        self.minimap = MinimapCalibration()
        self.settings = TrackerSettings()
        self.state = TrackerState()
        self._load_calibration()

    def _load_calibration(self):
        """Load saved calibration data if available."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    if 'minimap' in data:
                        self.minimap = MinimapCalibration(**data['minimap'])
            except (json.JSONDecodeError, TypeError):
                pass

    def save_calibration(self):
        """Save calibration data to file."""
        data = {
            'minimap': asdict(self.minimap)
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def reset_state(self):
        """Reset runtime state for new game."""
        self.state = TrackerState()


# Global config instance
config = Config()
