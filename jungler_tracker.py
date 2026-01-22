#!/usr/bin/env python3
"""
League of Legends Enemy Jungler Visibility Tracker
===================================================
A safe, passive overlay that displays whether the enemy jungler is visible
on the minimap using screen capture and image recognition.

Features:
- Green indicator: Enemy jungler is visible on the minimap
- Red indicator: Enemy jungler is NOT visible (be careful!)
- Works with any screen resolution
- Fully passive - no automated actions, just visual information

Requirements:
- Python 3.6+
- mss, opencv-python, numpy, Pillow, requests
- A League of Legends game in progress

Usage:
- Run this script while in a League of Legends game
- A small colored circle will appear in the top-left corner
- The overlay stays on top of all windows
"""

import tkinter as tk
import requests
import urllib3
import os
import io
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import mss
import cv2
import numpy as np
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Riot Live Client API endpoint (only accessible during an active game)
API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"

# Data Dragon base URL for champion images
DDRAGON_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{champion}.png"

# How often to check visibility (in milliseconds)
POLL_INTERVAL_MS = 200  # 200ms = 5 checks per second for responsive updates

# Overlay appearance settings
INDICATOR_SIZE = 40  # Size of the colored circle in pixels
INDICATOR_PADDING = 20  # Distance from screen edges in pixels

# Colors for the visibility indicator
COLOR_VISIBLE = "#00FF00"  # Bright green - jungler is visible
COLOR_INVISIBLE = "#FF0000"  # Bright red - jungler is NOT visible
COLOR_UNKNOWN = "#FFFF00"  # Yellow - waiting for game data or error
COLOR_NO_GAME = "#808080"  # Gray - no active game detected

# Minimap detection settings
# The minimap is in the bottom-right corner of the screen
# These ratios work for standard LoL UI at any resolution
MINIMAP_RATIO_WIDTH = 0.1425  # Minimap width as ratio of screen width
MINIMAP_RATIO_HEIGHT = 0.253  # Minimap height as ratio of screen height
MINIMAP_PADDING_RIGHT = 0  # Padding from right edge
MINIMAP_PADDING_BOTTOM = 0  # Padding from bottom edge

# Template matching threshold (0.0 to 1.0, higher = stricter matching)
MATCH_THRESHOLD = 0.55  # Lowered for better detection

# Champion icon size on minimap (ratio of minimap size)
ICON_SIZE_RATIO = 0.12  # Champion icons are about 12% of minimap size

# Cache directory for champion icons
CACHE_DIR = Path(__file__).parent / "champion_icons"

# Disable SSL warnings for the local Riot API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# DATA DRAGON / CHAMPION ICONS
# ============================================================================

def get_latest_ddragon_version() -> Optional[str]:
    """Get the latest Data Dragon version."""
    try:
        response = requests.get(DDRAGON_VERSION_URL, timeout=5.0)
        if response.status_code == 200:
            versions = response.json()
            return versions[0] if versions else None
    except Exception:
        pass
    return None


def download_champion_icon(champion_name: str, version: str) -> Optional[np.ndarray]:
    """
    Download a champion's square icon from Data Dragon.

    Returns the icon as an OpenCV image (BGR format).
    """
    # Normalize champion name for URL (handle special cases)
    url_name = champion_name.replace(" ", "").replace("'", "")

    # Special cases for champion names that differ in Data Dragon
    name_mappings = {
        "Wukong": "MonkeyKing",
        "Cho'Gath": "Chogath",
        "Kha'Zix": "Khazix",
        "Kai'Sa": "Kaisa",
        "Bel'Veth": "Belveth",
        "Vel'Koz": "Velkoz",
        "Rek'Sai": "RekSai",
        "Kog'Maw": "KogMaw",
        "LeBlanc": "Leblanc",
        "Nunu & Willump": "Nunu",
    }

    if champion_name in name_mappings:
        url_name = name_mappings[champion_name]

    url = DDRAGON_CHAMPION_URL.format(version=version, champion=url_name)

    try:
        response = requests.get(url, timeout=5.0)
        if response.status_code == 200:
            # Convert to OpenCV format
            image_data = io.BytesIO(response.content)
            pil_image = Image.open(image_data).convert("RGB")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return cv_image
    except Exception as e:
        print(f"Failed to download icon for {champion_name}: {e}")

    return None


def get_champion_icon(champion_name: str) -> Optional[np.ndarray]:
    """
    Get a champion icon, using cache if available.

    Returns the icon as an OpenCV image (BGR format).
    """
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)

    # Check cache first
    cache_file = CACHE_DIR / f"{champion_name}.png"

    if cache_file.exists():
        img = cv2.imread(str(cache_file))
        if img is not None:
            return img

    # Download from Data Dragon
    version = get_latest_ddragon_version()
    if version is None:
        print("Could not get Data Dragon version")
        return None

    icon = download_champion_icon(champion_name, version)

    if icon is not None:
        # Cache the icon
        cv2.imwrite(str(cache_file), icon)
        return icon

    return None


# ============================================================================
# MINIMAP DETECTION
# ============================================================================

class MinimapDetector:
    """
    Detects the minimap region and searches for champion icons on it.
    """

    def __init__(self):
        self.sct = mss.mss()
        self._update_screen_info()
        self.champion_template = None
        self.champion_name = None
        # Keep multiple scaled templates for better matching
        self.scaled_templates = []

    def _update_screen_info(self):
        """Update screen dimensions and minimap region."""
        # Get primary monitor info
        monitor = self.sct.monitors[1]  # monitors[0] is all monitors combined
        self.screen_width = monitor["width"]
        self.screen_height = monitor["height"]

        # Calculate minimap region based on screen size
        # The minimap is in the bottom-right corner
        minimap_width = int(self.screen_width * MINIMAP_RATIO_WIDTH)
        minimap_height = int(self.screen_height * MINIMAP_RATIO_HEIGHT)

        # Minimap position
        self.minimap_left = self.screen_width - minimap_width - MINIMAP_PADDING_RIGHT
        self.minimap_top = self.screen_height - minimap_height - MINIMAP_PADDING_BOTTOM
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height

        # Calculate expected icon size on minimap
        self.icon_size = int(min(minimap_width, minimap_height) * ICON_SIZE_RATIO)

        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print(f"Minimap region: ({self.minimap_left}, {self.minimap_top}) - {minimap_width}x{minimap_height}")
        print(f"Expected icon size: {self.icon_size}px")

    def set_champion(self, champion_name: str) -> bool:
        """
        Set the champion to search for on the minimap.

        Returns True if the champion icon was loaded successfully.
        """
        if champion_name == self.champion_name and self.champion_template is not None:
            return True  # Already loaded

        self.champion_name = champion_name
        icon = get_champion_icon(champion_name)

        if icon is None:
            print(f"Could not load icon for {champion_name}")
            self.champion_template = None
            self.scaled_templates = []
            return False

        # Create the main template at the expected icon size
        self.champion_template = self._prepare_template(icon, self.icon_size)

        # Create multiple scaled versions for better matching at different zoom levels
        self.scaled_templates = []
        for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            size = int(self.icon_size * scale)
            if size > 5:  # Minimum reasonable size
                template = self._prepare_template(icon, size)
                self.scaled_templates.append((scale, template))

        print(f"Loaded icon for {champion_name} ({len(self.scaled_templates)} scale variants)")
        return True

    def _prepare_template(self, icon: np.ndarray, size: int) -> np.ndarray:
        """
        Prepare a champion icon as a template for matching.

        The icon is resized and processed to improve matching accuracy.
        """
        # Resize to expected size on minimap
        template = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)

        # Create a circular mask (minimap icons are circular)
        mask = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        radius = int(size * 0.45)  # Slightly smaller than half to avoid edges
        cv2.circle(mask, (center, center), radius, 255, -1)

        # Apply mask to focus on the center of the icon
        template = cv2.bitwise_and(template, template, mask=mask)

        return template

    def capture_minimap(self) -> np.ndarray:
        """
        Capture the minimap region of the screen.

        Returns the minimap as an OpenCV image (BGR format).
        """
        # Define the region to capture
        region = {
            "left": self.minimap_left,
            "top": self.minimap_top,
            "width": self.minimap_width,
            "height": self.minimap_height
        }

        # Capture the region
        screenshot = self.sct.grab(region)

        # Convert to OpenCV format (mss returns BGRA)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def is_champion_visible(self) -> Tuple[bool, float]:
        """
        Check if the tracked champion is visible on the minimap.

        Returns a tuple of (is_visible, confidence).
        """
        if self.champion_template is None or not self.scaled_templates:
            return False, 0.0

        # Capture the minimap
        minimap = self.capture_minimap()

        # Try matching with each scaled template
        best_confidence = 0.0

        for scale, template in self.scaled_templates:
            # Skip if template is larger than minimap
            if template.shape[0] > minimap.shape[0] or template.shape[1] > minimap.shape[1]:
                continue

            # Perform template matching
            result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)

            # Get the best match
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val

        # Check if the best match exceeds our threshold
        is_visible = best_confidence >= MATCH_THRESHOLD

        return is_visible, best_confidence


# ============================================================================
# RIOT API INTERACTION
# ============================================================================

def get_game_data() -> Optional[Dict[str, Any]]:
    """Fetch all game data from the Riot Live Client API."""
    try:
        response = requests.get(API_URL, timeout=1.0, verify=False)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_active_player_team(game_data: Dict[str, Any]) -> Optional[str]:
    """Determine which team the active player is on."""
    try:
        active_player_name = game_data.get("activePlayer", {}).get("summonerName")
        if not active_player_name:
            return None

        for player in game_data.get("allPlayers", []):
            if player.get("summonerName") == active_player_name:
                return player.get("team")
    except (KeyError, TypeError):
        pass
    return None


def find_enemy_jungler(game_data: Dict[str, Any], my_team: str) -> Optional[Dict[str, Any]]:
    """Find the enemy jungler from the player list."""
    try:
        for player in game_data.get("allPlayers", []):
            if player.get("team") == my_team:
                continue
            if player.get("position", "").upper() == "JUNGLE":
                return player
    except (KeyError, TypeError):
        pass
    return None


def is_jungler_dead(jungler_data: Dict[str, Any]) -> bool:
    """Check if the jungler is dead or respawning."""
    is_dead = jungler_data.get("isDead", False)
    respawn_timer = jungler_data.get("respawnTimer", 0)
    return is_dead or respawn_timer > 0


# ============================================================================
# OVERLAY GUI
# ============================================================================

class JunglerTrackerOverlay:
    """
    A transparent overlay window that displays enemy jungler visibility.
    """

    def __init__(self):
        """Initialize the overlay window and start the update loop."""
        # Create the main window
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)

        try:
            self.root.attributes("-transparentcolor", "black")
            self.use_transparent_bg = True
        except tk.TclError:
            self.use_transparent_bg = False

        # Set window size and position
        self.root.geometry(f"{INDICATOR_SIZE}x{INDICATOR_SIZE}")
        self.root.geometry(f"+{INDICATOR_PADDING}+{INDICATOR_PADDING}")

        bg_color = "black" if self.use_transparent_bg else COLOR_NO_GAME
        self.root.configure(bg=bg_color)

        # Create the indicator canvas
        self.canvas = tk.Canvas(
            self.root,
            width=INDICATOR_SIZE,
            height=INDICATOR_SIZE,
            highlightthickness=0,
            bg=bg_color
        )
        self.canvas.pack()

        # Draw the initial indicator
        self.indicator = self.canvas.create_oval(
            2, 2,
            INDICATOR_SIZE - 2, INDICATOR_SIZE - 2,
            fill=COLOR_NO_GAME,
            outline="#000000",
            width=2
        )

        # State tracking
        self.current_color = COLOR_NO_GAME
        self.enemy_jungler_name = None

        # Initialize the minimap detector
        self.detector = MinimapDetector()

        # Make draggable
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Button-3>", self._close_overlay)

        # Start the update loop
        self._update_visibility()

        print("\nJungler Tracker Overlay started!")
        print("- Left-click and drag to move the overlay")
        print("- Right-click to close")
        print("- Waiting for game data...")

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _close_overlay(self, _event):
        print("Closing overlay...")
        self.root.destroy()

    def _update_indicator(self, color: str):
        if color != self.current_color:
            self.canvas.itemconfig(self.indicator, fill=color)
            self.current_color = color

    def _update_visibility(self):
        """Check visibility and update the indicator."""
        # Get game data from API
        game_data = get_game_data()

        if game_data is None:
            self._update_indicator(COLOR_NO_GAME)
        else:
            my_team = get_active_player_team(game_data)

            if my_team is None:
                self._update_indicator(COLOR_UNKNOWN)
            else:
                enemy_jungler = find_enemy_jungler(game_data, my_team)

                if enemy_jungler is None:
                    self._update_indicator(COLOR_UNKNOWN)
                else:
                    champion_name = enemy_jungler.get("championName", "Unknown")

                    # Update champion if changed
                    if champion_name != self.enemy_jungler_name:
                        self.enemy_jungler_name = champion_name
                        print(f"\nTracking enemy jungler: {champion_name}")
                        self.detector.set_champion(champion_name)

                    # Check if jungler is dead (always show green if dead)
                    if is_jungler_dead(enemy_jungler):
                        self._update_indicator(COLOR_VISIBLE)
                    else:
                        # Check minimap visibility
                        is_visible, confidence = self.detector.is_champion_visible()

                        if is_visible:
                            self._update_indicator(COLOR_VISIBLE)
                        else:
                            self._update_indicator(COLOR_INVISIBLE)

        # Schedule next update
        self.root.after(POLL_INTERVAL_MS, self._update_visibility)

    def run(self):
        """Start the overlay main loop."""
        self.root.mainloop()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 60)
    print("League of Legends Enemy Jungler Visibility Tracker")
    print("=" * 60)
    print()
    print("This overlay uses screen capture to detect if the enemy")
    print("jungler is visible on your minimap.")
    print()
    print("COLORS:")
    print("  GREEN  = Enemy jungler is visible on minimap (or dead)")
    print("  RED    = Enemy jungler NOT visible (be careful!)")
    print("  YELLOW = Error or no enemy jungler found")
    print("  GRAY   = No active game detected")
    print()

    # Check for required packages
    try:
        import mss
        import cv2
        import numpy
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return

    # Create and run the overlay
    overlay = JunglerTrackerOverlay()
    overlay.run()


if __name__ == "__main__":
    main()
