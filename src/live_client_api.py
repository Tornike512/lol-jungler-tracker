"""
Riot Live Client Data API Integration

This module interfaces with Riot's official Live Client Data API,
which runs on localhost:2999 during any League of Legends game.

Documentation: https://developer.riotgames.com/docs/lol#game-client-api

The API provides real-time game data without needing screen capture or OCR.
"""

import time
import requests
import urllib3
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Disable SSL warnings for localhost self-signed cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class PlayerStats:
    """Player statistics from Live Client API"""
    summoner_name: str
    champion_name: str
    level: int
    current_gold: float

    # CS (Creep Score)
    cs: int  # Total minions + monsters killed

    # Health and Mana
    current_health: float
    max_health: float
    current_mana: float  # Called "resource" in API
    max_mana: float

    # Position (if available)
    position_x: float = 0.0
    position_y: float = 0.0

    # Combat stats
    kills: int = 0
    deaths: int = 0
    assists: int = 0

    @property
    def health_percent(self) -> float:
        if self.max_health <= 0:
            return 0.0
        return self.current_health / self.max_health

    @property
    def mana_percent(self) -> float:
        if self.max_mana <= 0:
            return 1.0  # Some champs don't use mana
        return self.current_mana / self.max_mana


class LiveClientAPI:
    """
    Interface to Riot's Live Client Data API.

    The API is available at https://127.0.0.1:2999 during any game.
    Uses self-signed SSL certificate, so we disable SSL verification.
    """

    BASE_URL = "https://127.0.0.1:2999/liveclientdata"

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = False  # Self-signed cert

        # Caching
        self._last_data: Optional[Dict] = None
        self._last_fetch_time: float = 0.0
        self._cache_duration: float = 0.05  # 50ms cache (20 updates/sec max)

        # CS tracking
        self._last_cs: int = 0
        self._cs_history: list = []

        # Connection status
        self.connected: bool = False
        self._last_error: Optional[str] = None

    def is_game_running(self) -> bool:
        """Check if a game is currently running (API is available)"""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/allgamedata",
                timeout=self.timeout
            )
            self.connected = response.status_code == 200
            return self.connected
        except requests.exceptions.RequestException:
            self.connected = False
            return False

    def get_all_game_data(self) -> Optional[Dict[str, Any]]:
        """
        Get all game data in one request.

        Returns complete game state including:
        - activePlayer: Current player stats
        - allPlayers: All players in game
        - events: Game events
        - gameData: Game time, mode, etc.
        """
        current_time = time.time()

        # Return cached data if fresh
        if (self._last_data is not None and
            current_time - self._last_fetch_time < self._cache_duration):
            return self._last_data

        try:
            response = self.session.get(
                f"{self.BASE_URL}/allgamedata",
                timeout=self.timeout
            )

            if response.status_code == 200:
                self._last_data = response.json()
                self._last_fetch_time = current_time
                self.connected = True
                self._last_error = None
                return self._last_data
            else:
                self._last_error = f"HTTP {response.status_code}"
                self.connected = False
                return None

        except requests.exceptions.RequestException as e:
            self._last_error = str(e)
            self.connected = False
            return None

    def get_active_player(self) -> Optional[Dict[str, Any]]:
        """Get current player data only"""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/activeplayer",
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.connected = True
                return response.json()
        except requests.exceptions.RequestException:
            self.connected = False
        return None

    def get_player_stats(self) -> Optional[PlayerStats]:
        """
        Get current player statistics as a PlayerStats object.

        This is the main method you'll use for RL training.
        """
        data = self.get_all_game_data()
        if data is None:
            return None

        try:
            active = data.get("activePlayer", {})

            # Find current player in allPlayers for KDA and CS
            summoner_name = active.get("summonerName", "Unknown")
            all_players = data.get("allPlayers", [])

            player_data = None
            for player in all_players:
                if player.get("summonerName") == summoner_name:
                    player_data = player
                    break

            # Get CS from player data (scores.creepScore)
            scores = player_data.get("scores", {}) if player_data else {}
            cs = scores.get("creepScore", 0)
            kills = scores.get("kills", 0)
            deaths = scores.get("deaths", 0)
            assists = scores.get("assists", 0)

            # Get champion stats
            champion_stats = active.get("championStats", {})

            return PlayerStats(
                summoner_name=summoner_name,
                champion_name=active.get("championName", "Unknown"),
                level=active.get("level", 1),
                current_gold=active.get("currentGold", 0.0),
                cs=cs,
                current_health=champion_stats.get("currentHealth", 100.0),
                max_health=champion_stats.get("maxHealth", 100.0),
                current_mana=champion_stats.get("resourceValue", 100.0),
                max_mana=champion_stats.get("resourceMax", 100.0),
                kills=kills,
                deaths=deaths,
                assists=assists
            )

        except (KeyError, TypeError) as e:
            self._last_error = f"Parse error: {e}"
            return None

    def get_cs(self) -> tuple[int, int]:
        """
        Get current CS and CS gained since last call.

        Returns:
            (current_cs, cs_gained) tuple
        """
        stats = self.get_player_stats()
        if stats is None:
            return self._last_cs, 0

        current_cs = stats.cs
        cs_gained = max(0, current_cs - self._last_cs)

        # Track history
        self._cs_history.append({
            "time": time.time(),
            "cs": current_cs
        })
        if len(self._cs_history) > 100:
            self._cs_history.pop(0)

        self._last_cs = current_cs
        return current_cs, cs_gained

    def get_cs_per_minute(self) -> float:
        """Calculate CS per minute from history"""
        if len(self._cs_history) < 2:
            return 0.0

        first = self._cs_history[0]
        last = self._cs_history[-1]

        time_diff = last["time"] - first["time"]
        cs_diff = last["cs"] - first["cs"]

        if time_diff <= 0:
            return 0.0

        return (cs_diff / time_diff) * 60.0

    def get_game_time(self) -> float:
        """Get current game time in seconds"""
        data = self.get_all_game_data()
        if data is None:
            return 0.0

        game_data = data.get("gameData", {})
        return game_data.get("gameTime", 0.0)

    def get_events(self) -> list:
        """Get list of game events (kills, objectives, etc.)"""
        data = self.get_all_game_data()
        if data is None:
            return []

        events = data.get("events", {})
        return events.get("Events", [])

    def reset(self):
        """Reset CS tracking for new episode"""
        self._last_cs = 0
        self._cs_history = []
        self._last_data = None

    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics for debugging"""
        return {
            "connected": self.connected,
            "last_error": self._last_error,
            "current_cs": self._last_cs,
            "cs_per_minute": self.get_cs_per_minute(),
            "cache_age_ms": (time.time() - self._last_fetch_time) * 1000 if self._last_fetch_time > 0 else 0
        }


# Convenience wrapper that mimics CSDetector interface
class LiveClientCSDetector:
    """
    Drop-in replacement for CSDetector that uses Live Client API.

    Same interface as CSDetector but faster and more accurate.
    """

    def __init__(self, **kwargs):
        # Ignore CSDetector parameters (screen_width, etc.)
        self.api = LiveClientAPI()
        self.current_cs = 0
        self._last_cs = 0

        print("LiveClientCSDetector initialized")
        print("  Using Riot Live Client API (localhost:2999)")
        print("  Waiting for game to start...")

    def update(self) -> tuple[int, int]:
        """
        Update CS reading from Live Client API.

        Returns:
            (current_cs, cs_gained) tuple
        """
        current_cs, cs_gained = self.api.get_cs()
        self.current_cs = current_cs
        return current_cs, cs_gained

    def get_cs_per_minute(self) -> float:
        """Get CS per minute"""
        return self.api.get_cs_per_minute()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = self.api.get_stats()
        return {
            "current_cs": stats["current_cs"],
            "cs_per_minute": stats["cs_per_minute"],
            "successful_reads": 1 if stats["connected"] else 0,
            "failed_reads": 0 if stats["connected"] else 1,
            "ocr_time_ms": stats["cache_age_ms"],  # Not really OCR but similar metric
            "accuracy": 1.0 if stats["connected"] else 0.0,
            "api_connected": stats["connected"],
            "api_error": stats["last_error"]
        }

    def reset(self):
        """Reset CS tracking"""
        self.api.reset()
        self.current_cs = 0
        self._last_cs = 0


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Live Client API")
    print("=" * 60)
    print("Make sure League of Legends is running with an active game")
    print()

    api = LiveClientAPI()

    # Check if game is running
    print("Checking for active game...")
    if not api.is_game_running():
        print("No game detected. Start a game and try again.")
        print("(The API is only available during active gameplay)")
        exit(1)

    print("Game detected! Fetching data...")
    print()

    # Get player stats
    stats = api.get_player_stats()
    if stats:
        print(f"Champion: {stats.champion_name}")
        print(f"Level: {stats.level}")
        print(f"CS: {stats.cs}")
        print(f"Gold: {stats.current_gold:.0f}")
        print(f"HP: {stats.current_health:.0f}/{stats.max_health:.0f} ({stats.health_percent:.1%})")
        print(f"Mana: {stats.current_mana:.0f}/{stats.max_mana:.0f} ({stats.mana_percent:.1%})")
        print(f"KDA: {stats.kills}/{stats.deaths}/{stats.assists}")
    else:
        print("Could not fetch player stats")
        print(f"Error: {api._last_error}")

    print()
    print(f"Game time: {api.get_game_time():.1f} seconds")

    # Monitor CS changes
    print()
    print("Monitoring CS for 30 seconds...")
    print("Try last-hitting some minions!")
    print()

    start_time = time.time()
    last_print_time = 0

    try:
        while time.time() - start_time < 30:
            current_cs, cs_gained = api.get_cs()

            if cs_gained > 0:
                print(f"  >>> +{cs_gained} CS! (Total: {current_cs})")

            # Print every second
            if time.time() - last_print_time >= 1.0:
                stats = api.get_stats()
                print(f"CS: {current_cs} | CS/min: {stats['cs_per_minute']:.1f}")
                last_print_time = time.time()

            time.sleep(0.1)  # 10 Hz polling

    except KeyboardInterrupt:
        print("\nStopped by user")

    print()
    print("Final stats:")
    for key, value in api.get_stats().items():
        print(f"  {key}: {value}")
