"""
Champion icon detection and visibility tracking.
Detects enemy jungler on the minimap using template matching.
Also provides auto-detection via Riot Live Client API.
"""
import numpy as np
import cv2
import os
import time
import urllib.request
import json
import ssl
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from config import config, CHAMPIONS_DIR
from zones import get_zone_at_position, normalize_position, Zone


# Riot Live Client API settings
RIOT_API_BASE = "https://127.0.0.1:2999/liveclientdata"


def get_game_data() -> Optional[Dict[str, Any]]:
    """
    Fetch game data from Riot Live Client API.

    Returns:
        Game data dictionary or None if unavailable
    """
    try:
        # Riot's local API uses a self-signed certificate
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        url = f"{RIOT_API_BASE}/allgamedata"
        req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=2, context=ctx) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        # API not available (game not running or not in-game)
        return None


def detect_enemy_jungler() -> Optional[str]:
    """
    Auto-detect the enemy jungler using Riot Live Client API.

    Finds the enemy player with Smite summoner spell.

    Returns:
        Champion name of enemy jungler, or None if not found
    """
    data = get_game_data()
    if data is None:
        return None

    try:
        # Get the active player's team
        active_player = data.get("activePlayer", {})
        active_summoner = active_player.get("summonerName", "")

        # Find active player's team
        all_players = data.get("allPlayers", [])
        player_team = None

        for player in all_players:
            if player.get("summonerName") == active_summoner:
                player_team = player.get("team")
                break

        if player_team is None:
            # Fallback: try to determine from riotId
            riot_id = active_player.get("riotId", "")
            for player in all_players:
                if player.get("riotId") == riot_id:
                    player_team = player.get("team")
                    break

        if player_team is None:
            print("[WARNING] Could not determine player team")
            return None

        # Find enemy with Smite
        for player in all_players:
            # Skip players on our team
            if player.get("team") == player_team:
                continue

            # Check summoner spells for Smite
            spells = player.get("summonerSpells", {})
            spell_one = spells.get("summonerSpellOne", {}).get("displayName", "")
            spell_two = spells.get("summonerSpellTwo", {}).get("displayName", "")

            if "Smite" in spell_one or "Smite" in spell_two:
                champion = player.get("championName", "")
                if champion:
                    print(f"[STATUS] Auto-detected enemy jungler: {champion}")
                    return champion

        print("[WARNING] No enemy with Smite found")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to parse game data: {e}")
        return None


def get_all_enemy_champions() -> List[str]:
    """
    Get list of all enemy champion names.

    Returns:
        List of enemy champion names
    """
    data = get_game_data()
    if data is None:
        return []

    try:
        active_player = data.get("activePlayer", {})
        active_summoner = active_player.get("summonerName", "")

        all_players = data.get("allPlayers", [])
        player_team = None

        for player in all_players:
            if player.get("summonerName") == active_summoner:
                player_team = player.get("team")
                break

        if player_team is None:
            riot_id = active_player.get("riotId", "")
            for player in all_players:
                if player.get("riotId") == riot_id:
                    player_team = player.get("team")
                    break

        enemies = []
        for player in all_players:
            if player.get("team") != player_team:
                champion = player.get("championName", "")
                if champion:
                    enemies.append(champion)

        return enemies

    except Exception:
        return []


def is_game_active() -> bool:
    """
    Check if a League of Legends game is currently active.

    Returns:
        True if in-game, False otherwise
    """
    return get_game_data() is not None


@dataclass
class Detection:
    """Represents a champion detection on the minimap."""
    champion: str
    position: Tuple[float, float]  # Normalized (0-1)
    pixel_position: Tuple[int, int]  # Raw pixels on minimap
    zone: Optional[Zone]
    confidence: float
    timestamp: float
    is_visible: bool


class ChampionDetector:
    """Detects champion icons on the minimap."""

    # Data Dragon base URL for champion icons
    DDRAGON_VERSION = "14.1.1"  # Update as needed
    DDRAGON_URL = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion"

    def __init__(self):
        self._templates: dict = {}  # Champion name -> list of template images
        self._current_champion: str = ""
        self._last_detection: Optional[Detection] = None
        self._visibility_history: List[bool] = []
        self._history_size = 5  # Track last N frames for stability

    def set_champion(self, champion_name: str) -> bool:
        """
        Set the champion to track and load/download icon templates.

        Args:
            champion_name: Name of the champion (e.g., "Lee Sin", "Elise")

        Returns:
            True if templates loaded successfully
        """
        # Normalize champion name (remove spaces, capitalize correctly)
        normalized = self._normalize_champion_name(champion_name)
        self._current_champion = normalized

        # Try to load existing templates
        if self._load_templates(normalized):
            print(f"[STATUS] Loaded templates for {normalized}")
            return True

        # Try to download from Data Dragon
        if self._download_champion_icon(normalized):
            return self._load_templates(normalized)

        print(f"[WARNING] Could not load templates for {normalized}. "
              "Place icon images in assets/champions/{normalized}/")
        return False

    def _normalize_champion_name(self, name: str) -> str:
        """Normalize champion name to Data Dragon format."""
        # Handle special cases
        special_cases = {
            "lee sin": "LeeSin",
            "master yi": "MasterYi",
            "miss fortune": "MissFortune",
            "twisted fate": "TwistedFate",
            "dr. mundo": "DrMundo",
            "dr mundo": "DrMundo",
            "jarvan iv": "JarvanIV",
            "jarvan": "JarvanIV",
            "aurelion sol": "AurelionSol",
            "tahm kench": "TahmKench",
            "rek'sai": "RekSai",
            "reksai": "RekSai",
            "rek sai": "RekSai",
            "kha'zix": "Khazix",
            "khazix": "Khazix",
            "kha zix": "Khazix",
            "cho'gath": "Chogath",
            "chogath": "Chogath",
            "cho gath": "Chogath",
            "vel'koz": "Velkoz",
            "velkoz": "Velkoz",
            "xin zhao": "XinZhao",
            "nunu & willump": "Nunu",
            "nunu and willump": "Nunu",
            "nunu": "Nunu",
            "wukong": "MonkeyKing",
            "renata glasc": "Renata",
            "bel'veth": "Belveth",
            "kai'sa": "Kaisa",
            "k'sante": "KSante",
        }

        lower_name = name.lower().strip()
        if lower_name in special_cases:
            return special_cases[lower_name]

        # Default: remove spaces and apostrophes, capitalize first letter of each word
        return "".join(word.capitalize() for word in name.replace("'", "").split())

    def _load_templates(self, champion: str) -> bool:
        """Load template images for a champion."""
        champion_dir = os.path.join(CHAMPIONS_DIR, champion)
        if not os.path.exists(champion_dir):
            os.makedirs(champion_dir, exist_ok=True)
            return False

        templates = []
        for filename in os.listdir(champion_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(champion_dir, filename)
                template = cv2.imread(filepath)
                if template is not None:
                    templates.append(template)
                    # Also add scaled versions for different minimap sizes
                    for scale in [0.8, 0.9, 1.1, 1.2]:
                        scaled = cv2.resize(template, None, fx=scale, fy=scale)
                        templates.append(scaled)

        if templates:
            self._templates[champion] = templates
            return True
        return False

    def _download_champion_icon(self, champion: str) -> bool:
        """Download champion icon from Data Dragon."""
        try:
            url = f"{self.DDRAGON_URL}/{champion}.png"
            champion_dir = os.path.join(CHAMPIONS_DIR, champion)
            os.makedirs(champion_dir, exist_ok=True)

            filepath = os.path.join(champion_dir, f"{champion}.png")
            print(f"[STATUS] Downloading icon for {champion}...")

            urllib.request.urlretrieve(url, filepath)
            print(f"[STATUS] Downloaded {champion} icon")

            # Create minimap-sized version (champion icons on minimap are ~20-30px)
            icon = cv2.imread(filepath)
            if icon is not None:
                # Create multiple size variants
                for size in [16, 20, 24, 28, 32]:
                    resized = cv2.resize(icon, (size, size))
                    size_path = os.path.join(champion_dir, f"{champion}_{size}.png")
                    cv2.imwrite(size_path, resized)

            return True
        except Exception as e:
            print(f"[ERROR] Failed to download icon: {e}")
            return False

    def detect(self, minimap: np.ndarray) -> Optional[Detection]:
        """
        Detect the tracked champion on the minimap.

        Args:
            minimap: BGR image of the minimap

        Returns:
            Detection object if found, None otherwise
        """
        if minimap is None or not self._current_champion:
            return None

        templates = self._templates.get(self._current_champion, [])
        if not templates:
            return None

        best_match = None
        best_confidence = 0.0
        best_location = None

        minimap_height, minimap_width = minimap.shape[:2]

        for template in templates:
            if template.shape[0] > minimap_height or template.shape[1] > minimap_width:
                continue

            # Template matching
            result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence and max_val > config.settings.template_threshold:
                best_confidence = max_val
                best_location = max_loc
                best_match = template

        current_time = time.time()

        if best_match is not None and best_location is not None:
            # Calculate center of detected icon
            template_h, template_w = best_match.shape[:2]
            center_x = best_location[0] + template_w // 2
            center_y = best_location[1] + template_h // 2

            # Normalize position
            norm_x, norm_y = normalize_position(
                center_x, center_y, minimap_width, minimap_height
            )

            # Get zone
            zone = get_zone_at_position(norm_x, norm_y)

            detection = Detection(
                champion=self._current_champion,
                position=(norm_x, norm_y),
                pixel_position=(center_x, center_y),
                zone=zone,
                confidence=best_confidence,
                timestamp=current_time,
                is_visible=True
            )

            self._update_visibility_history(True)
            self._last_detection = detection
            return detection

        # No detection - jungler not visible
        self._update_visibility_history(False)

        # Return last known position with is_visible=False
        if self._last_detection is not None:
            return Detection(
                champion=self._last_detection.champion,
                position=self._last_detection.position,
                pixel_position=self._last_detection.pixel_position,
                zone=self._last_detection.zone,
                confidence=0.0,
                timestamp=current_time,
                is_visible=False
            )

        return None

    def _update_visibility_history(self, is_visible: bool):
        """Track visibility over multiple frames for stability."""
        self._visibility_history.append(is_visible)
        if len(self._visibility_history) > self._history_size:
            self._visibility_history.pop(0)

    def just_disappeared(self) -> bool:
        """
        Check if the jungler just transitioned from visible to invisible.
        Uses history to avoid false triggers from single-frame detection failures.
        """
        if len(self._visibility_history) < 3:
            return False

        # Check if we had visibility recently but now don't
        recent = self._visibility_history[-3:]
        # Pattern: [True, True/False, False] - was visible, now gone
        return recent[0] and not recent[-1]

    def just_appeared(self) -> bool:
        """Check if the jungler just became visible."""
        if len(self._visibility_history) < 3:
            return False

        recent = self._visibility_history[-3:]
        # Pattern: [False, False/True, True] - wasn't visible, now is
        return not recent[0] and recent[-1]

    def is_stable_visible(self) -> bool:
        """Check if jungler has been consistently visible."""
        if len(self._visibility_history) < 3:
            return False
        return all(self._visibility_history[-3:])

    def is_stable_invisible(self) -> bool:
        """Check if jungler has been consistently invisible."""
        if len(self._visibility_history) < 3:
            return False
        return not any(self._visibility_history[-3:])

    def get_last_known_position(self) -> Optional[Tuple[float, float]]:
        """Get the last known position of the jungler."""
        if self._last_detection:
            return self._last_detection.position
        return None

    def get_last_known_zone(self) -> Optional[Zone]:
        """Get the last known zone of the jungler."""
        if self._last_detection:
            return self._last_detection.zone
        return None

    def reset(self):
        """Reset detector state for new game."""
        self._last_detection = None
        self._visibility_history = []


# Global detector instance
champion_detector = ChampionDetector()
