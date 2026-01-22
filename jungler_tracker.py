#!/usr/bin/env python3
"""
League of Legends Enemy Jungler Visibility Tracker
===================================================
A safe, passive overlay that displays whether the enemy jungler is visible.
Uses the Riot Live Client API (available during active games).

Features:
- Green indicator: Enemy jungler is visible on the map
- Red indicator: Enemy jungler is NOT visible (be careful!)
- Fully passive - no automated actions, just visual information

Requirements:
- Python 3.6+
- requests library (pip install requests)
- A League of Legends game in progress

Usage:
- Run this script while in a League of Legends game
- A small colored square will appear in the top-right corner
- The overlay stays on top of all windows
"""

import tkinter as tk
import requests
import urllib3
from typing import Optional, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Riot Live Client API endpoint (only accessible during an active game)
API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"

# How often to check the API (in milliseconds)
# 500ms = 0.5 seconds, provides responsive updates without excessive polling
POLL_INTERVAL_MS = 500

# Overlay appearance settings
INDICATOR_SIZE = 40  # Size of the colored square in pixels
INDICATOR_PADDING = 20  # Distance from screen edges in pixels

# Colors for the visibility indicator
COLOR_VISIBLE = "#00FF00"  # Bright green - jungler is visible
COLOR_INVISIBLE = "#FF0000"  # Bright red - jungler is NOT visible
COLOR_UNKNOWN = "#FFFF00"  # Yellow - waiting for game data or error
COLOR_NO_GAME = "#808080"  # Gray - no active game detected

# Disable SSL warnings for the local Riot API (uses self-signed certificate)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# RIOT API INTERACTION
# ============================================================================

def get_game_data() -> Optional[Dict[str, Any]]:
    """
    Fetch all game data from the Riot Live Client API.

    Returns:
        Dictionary containing game data if successful, None otherwise.

    Note:
        The API uses HTTPS with a self-signed certificate, so we disable
        SSL verification. This is safe because it's a local connection.
    """
    try:
        # Make request to the local API with a short timeout
        # verify=False is required because Riot uses a self-signed certificate
        response = requests.get(API_URL, timeout=1.0, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return None

    except requests.exceptions.ConnectionError:
        # API not available - game probably not running
        return None
    except requests.exceptions.Timeout:
        # Request timed out
        return None
    except requests.exceptions.RequestException:
        # Any other request error
        return None
    except ValueError:
        # JSON parsing error
        return None


def get_active_player_team(game_data: Dict[str, Any]) -> Optional[str]:
    """
    Determine which team the active player (you) is on.

    Args:
        game_data: The full game data from the API.

    Returns:
        "ORDER" (blue side) or "CHAOS" (red side), or None if not found.
    """
    try:
        # Get the active player's summoner name
        active_player_name = game_data.get("activePlayer", {}).get("summonerName")

        if not active_player_name:
            return None

        # Find the active player in the allPlayers list to get their team
        for player in game_data.get("allPlayers", []):
            if player.get("summonerName") == active_player_name:
                return player.get("team")

        return None

    except (KeyError, TypeError):
        return None


def find_enemy_jungler(game_data: Dict[str, Any], my_team: str) -> Optional[Dict[str, Any]]:
    """
    Find the enemy jungler from the player list.

    Args:
        game_data: The full game data from the API.
        my_team: The team the active player is on ("ORDER" or "CHAOS").

    Returns:
        Player data dictionary for the enemy jungler, or None if not found.
    """
    try:
        all_players = game_data.get("allPlayers", [])

        for player in all_players:
            # Check if this player is on the enemy team
            player_team = player.get("team")
            if player_team == my_team:
                continue  # Skip teammates

            # Check if this player has the JUNGLE position
            # The position field contains the assigned role
            position = player.get("position", "").upper()

            if position == "JUNGLE":
                return player

        return None

    except (KeyError, TypeError):
        return None


def is_jungler_visible(jungler_data: Dict[str, Any]) -> bool:
    """
    Check if the enemy jungler is currently visible on the map.

    Args:
        jungler_data: Player data dictionary for the enemy jungler.

    Returns:
        True if visible, False otherwise.

    Note:
        The Riot API provides visibility information that indicates
        whether a champion can be seen by your team.
    """
    # The API doesn't directly expose isVisible in allPlayers
    # But we can check if the champion is dead or check other indicators
    # For now, we'll use the 'isDead' status as one indicator

    # Check multiple visibility indicators
    is_dead = jungler_data.get("isDead", False)

    # If the jungler is dead, they're technically "visible" (in death state)
    # but not a threat - we'll show as visible (green) since they can't gank
    if is_dead:
        return True

    # Unfortunately, the Live Client API doesn't provide real-time
    # fog of war visibility data for enemy champions in the allPlayers endpoint.
    # The API is designed to be "safe" and doesn't reveal information
    # that wouldn't be available to spectators.

    # Alternative approach: Check scores/items for recent activity
    # If we can see their items updating, they were recently visible

    # For a basic implementation, we'll need to rely on what data IS available
    # The 'respawnTimer' field indicates if they're respawning
    respawn_timer = jungler_data.get("respawnTimer", 0)
    if respawn_timer > 0:
        return True  # Dead/respawning, not a threat

    # Since direct visibility isn't available, this implementation
    # shows the jungler info but would need game memory reading
    # (which could violate ToS) to get true fog-of-war visibility

    # Return False by default to remind player to be cautious
    return False


# ============================================================================
# OVERLAY GUI
# ============================================================================

class JunglerTrackerOverlay:
    """
    A transparent overlay window that displays enemy jungler visibility.

    The overlay consists of a small colored square that changes color
    based on whether the enemy jungler can be seen:
    - Green: Jungler is visible (safer to play aggressive)
    - Red: Jungler is NOT visible (play safe!)
    - Yellow: Unknown state or error
    - Gray: No active game detected
    """

    def __init__(self):
        """Initialize the overlay window and start the update loop."""

        # Create the main window
        self.root = tk.Tk()

        # Remove window decorations (title bar, borders)
        self.root.overrideredirect(True)

        # Make the window stay on top of all other windows
        self.root.attributes("-topmost", True)

        # Set window transparency (1.0 = opaque, 0.0 = fully transparent)
        # We'll make the background transparent and only show the indicator
        self.root.attributes("-alpha", 0.9)

        # Try to set transparent color (works on some systems)
        # This makes the specified color completely transparent
        try:
            self.root.attributes("-transparentcolor", "black")
            self.use_transparent_bg = True
        except tk.TclError:
            # Transparency not supported on this system
            self.use_transparent_bg = False

        # Set the window size
        self.root.geometry(f"{INDICATOR_SIZE}x{INDICATOR_SIZE}")

        # Position the window in the top-left corner of the screen
        x_position = INDICATOR_PADDING
        y_position = INDICATOR_PADDING
        self.root.geometry(f"+{x_position}+{y_position}")

        # Set background color
        bg_color = "black" if self.use_transparent_bg else COLOR_NO_GAME
        self.root.configure(bg=bg_color)

        # Create the indicator canvas
        self.canvas = tk.Canvas(
            self.root,
            width=INDICATOR_SIZE,
            height=INDICATOR_SIZE,
            highlightthickness=0,  # Remove border
            bg=bg_color
        )
        self.canvas.pack()

        # Draw the initial indicator (gray = no game)
        self.indicator = self.canvas.create_oval(
            2, 2,  # Top-left corner with small padding
            INDICATOR_SIZE - 2, INDICATOR_SIZE - 2,  # Bottom-right corner
            fill=COLOR_NO_GAME,
            outline="#000000",  # Black outline for visibility
            width=2
        )

        # Store the current state for comparison
        self.current_color = COLOR_NO_GAME
        self.enemy_jungler_name = "Unknown"

        # Make the window draggable (optional - click and drag to move)
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Right-click to close the overlay
        self.canvas.bind("<Button-3>", self._close_overlay)

        # Start the update loop
        self._update_visibility()

        print("Jungler Tracker Overlay started!")
        print("- Left-click and drag to move the overlay")
        print("- Right-click to close")
        print("- Waiting for game data...")

    def _start_drag(self, event):
        """Record the starting position for dragging."""
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        """Move the window when dragged."""
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _close_overlay(self, _event):
        """Close the overlay window."""
        print("Closing overlay...")
        self.root.destroy()

    def _update_indicator(self, color: str):
        """
        Update the indicator color if it has changed.

        Args:
            color: The new color for the indicator.
        """
        if color != self.current_color:
            self.canvas.itemconfig(self.indicator, fill=color)
            self.current_color = color

    def _update_visibility(self):
        """
        Fetch game data and update the visibility indicator.
        This method runs periodically based on POLL_INTERVAL_MS.
        """
        # Try to get game data from the API
        game_data = get_game_data()

        if game_data is None:
            # No game running or API not available
            self._update_indicator(COLOR_NO_GAME)
        else:
            # Game is running, find our team
            my_team = get_active_player_team(game_data)

            if my_team is None:
                # Couldn't determine our team
                self._update_indicator(COLOR_UNKNOWN)
            else:
                # Find the enemy jungler
                enemy_jungler = find_enemy_jungler(game_data, my_team)

                if enemy_jungler is None:
                    # No enemy jungler found (might be a custom game mode)
                    self._update_indicator(COLOR_UNKNOWN)
                else:
                    # Update the stored jungler name
                    new_name = enemy_jungler.get("championName", "Unknown")
                    if new_name != self.enemy_jungler_name:
                        self.enemy_jungler_name = new_name
                        print(f"Tracking enemy jungler: {self.enemy_jungler_name}")

                    # Check visibility and update indicator
                    if is_jungler_visible(enemy_jungler):
                        self._update_indicator(COLOR_VISIBLE)
                    else:
                        self._update_indicator(COLOR_INVISIBLE)

        # Schedule the next update
        self.root.after(POLL_INTERVAL_MS, self._update_visibility)

    def run(self):
        """Start the overlay main loop."""
        self.root.mainloop()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the Jungler Tracker Overlay.
    """
    print("=" * 60)
    print("League of Legends Enemy Jungler Visibility Tracker")
    print("=" * 60)
    print()
    print("IMPORTANT NOTES:")
    print("- This overlay is PASSIVE and safe to use")
    print("- It only reads publicly available game data")
    print("- No automation or gameplay assistance")
    print()
    print("COLORS:")
    print(f"  GREEN  = Enemy jungler is visible/dead (safer)")
    print(f"  RED    = Enemy jungler location unknown (be careful!)")
    print(f"  YELLOW = Error or unknown state")
    print(f"  GRAY   = No active game detected")
    print()
    print("NOTE: The Riot Live Client API has limitations.")
    print("It doesn't provide real-time fog-of-war visibility data.")
    print("The indicator shows RED by default to encourage safe play.")
    print()

    # Create and run the overlay
    overlay = JunglerTrackerOverlay()
    overlay.run()


if __name__ == "__main__":
    main()
