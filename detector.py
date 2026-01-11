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


RIOT_API_BASE = "https://127.0.0.1:2999/liveclientdata"


def get_game_data():
    """Fetch game data from Riot Live Client API."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = RIOT_API_BASE + "/allgamedata"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2, context=ctx) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None


def detect_enemy_jungler():
    """Auto-detect the enemy jungler using Riot Live Client API."""
    data = get_game_data()
    if data is None:
        return None

    try:
        active_player = data.get("activePlayer", {})
        active_summoner = active_player. get("summonerName", "")
        all_players = data.get("allPlayers", [])
        player_team = None

        for player in all_players:
            if player.get("summonerName") == active_summoner:
                player_team = player.get("team")
                break

        if player_team is None:
            riot_id = active_player.get("riotId", "")
            for player in all_players:
                if player. get("riotId") == riot_id:
                    player_team = player.get("team")
                    break

        if player_team is None:
            print("[WARNING] Could not determine player team")
            return None

        for player in all_players:
            if player.get("team") == player_team:
                continue
            spells = player.get("summonerSpells", {})
            spell_one = spells.get("summonerSpellOne", {}
                                   ).get("displayName", "")
            spell_two = spells.get("summonerSpellTwo", {}
                                   ).get("displayName", "")
            if "Smite" in spell_one or "Smite" in spell_two:
                champion = player.get("championName", "")
                if champion:
                    print("[STATUS] Auto-detected enemy jungler: " + champion)
                    return champion

        print("[WARNING] No enemy with Smite found")
        return None

    except Exception as e:
        print("[ERROR] Failed to parse game data: " + str(e))
        return None


def get_all_enemy_champions():
    """Get list of all enemy champion names."""
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
                player_team = player. get("team")
                break

        if player_team is None:
            riot_id = active_player.get("riotId", "")
            for player in all_players:
                if player. get("riotId") == riot_id:
                    player_team = player.get("team")
                    break

        enemies = []
        for player in all_players:
            if player. get("team") != player_team:
                champion = player.get("championName", "")
                if champion:
                    enemies.append(champion)

        return enemies

    except Exception:
        return []


def is_game_active():
    """Check if a League of Legends game is currently active."""
    return get_game_data() is not None


@dataclass
class Detection:
    """Represents a champion detection on the minimap."""
    champion: str
    position: Tuple[float, float]
    pixel_position: Tuple[int, int]
    zone: Optional[Zone]
    confidence: float
    timestamp: float
    is_visible: bool


class ChampionDetector:
    """Detects champion icons on the minimap."""

    DDRAGON_VERSION = "14.1.1"
    DDRAGON_URL = "https://ddragon.leagueoflegends.com/cdn/14.1.1/img/champion"

    def __init__(self):
        self._templates = {}
        self._current_champion = ""
        self._last_detection = None
        self._visibility_history = []
        self._history_size = 5
        self._match_threshold = 0.85

    def set_champion(self, champion_name):
        """Set the champion to track and load/download icon templates."""
        normalized = self._normalize_champion_name(champion_name)
        self._current_champion = normalized

        if self._load_templates(normalized):
            print("[STATUS] Loaded templates for " + normalized)
            return True

        if self._download_champion_icon(normalized):
            return self._load_templates(normalized)

        print("[WARNING] Could not load templates for " + normalized)
        return False

    def _normalize_champion_name(self, name):
        """Normalize champion name to Data Dragon format."""
        special_cases = {
            "lee sin": "LeeSin",
            "master yi": "MasterYi",
            "miss fortune": "MissFortune",
            "twisted fate": "TwistedFate",
            "dr. mundo": "DrMundo",
            "dr mundo": "DrMundo",
            "jarvan iv": "JarvanIV",
            "jarvan":  "JarvanIV",
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
            "cho gath":  "Chogath",
            "vel'koz": "Velkoz",
            "velkoz": "Velkoz",
            "xin zhao": "XinZhao",
            "nunu & willump": "Nunu",
            "nunu and willump": "Nunu",
            "nunu":  "Nunu",
            "wukong": "MonkeyKing",
            "renata glasc": "Renata",
            "bel'veth": "Belveth",
            "kai'sa": "Kaisa",
            "k'sante": "KSante",
        }

        lower_name = name.lower().strip()
        if lower_name in special_cases:
            return special_cases[lower_name]

        return "".join(word.capitalize() for word in name.replace("'", "").split())

    def _load_templates(self, champion):
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
                    template = self._crop_center(template)
                    templates.append(template)
                    for scale in [0.8, 0.9, 1.1, 1.2]:
                        scaled = cv2.resize(template, None, fx=scale, fy=scale)
                        templates.append(scaled)

        if templates:
            self._templates[champion] = templates
            return True
        return False

    def _crop_center(self, image, crop_percent=0.15):
        """Crop the center of an image to remove border area."""
        h, w = image.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)

        if crop_h * 2 >= h or crop_w * 2 >= w:
            return image

        return image[crop_h: h-crop_h, crop_w:w-crop_w]

    def _download_champion_icon(self, champion):
        """Download champion icon from Data Dragon."""
        try:
            url = self.DDRAGON_URL + "/" + champion + ".png"
            champion_dir = os.path.join(CHAMPIONS_DIR, champion)
            os.makedirs(champion_dir, exist_ok=True)

            filepath = os.path.join(champion_dir, champion + ".png")
            print("[STATUS] Downloading icon for " + champion + "...")
            print("[DEBUG] URL: " + url)

            urllib.request.urlretrieve(url, filepath)
            print("[STATUS] Downloaded " + champion + " icon")

            icon = cv2.imread(filepath)
            if icon is not None:
                for size in [16, 20, 24, 28, 32]:
                    resized = cv2.resize(icon, (size, size))
                    size_path = os.path.join(
                        champion_dir, champion + "_" + str(size) + ".png")
                    cv2.imwrite(size_path, resized)

            return True
        except Exception as e:
            print("[ERROR] Failed to download icon:  " + str(e))
            return False

    def _filter_red_borders(self, minimap):
        """Filter out strong red areas to reduce false positives."""
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        minimap_filtered = minimap.copy()
        minimap_filtered[red_mask > 0] = [128, 128, 128]

        return minimap_filtered

    def _is_valid_champion_location(self, x, y, minimap_width, minimap_height):
        """Check if the detected location is a valid champion position."""
        norm_x = x / minimap_width
        norm_y = y / minimap_height

        tower_positions = [
            (0.05, 0.95, 0.08),
            (0.15, 0.85, 0.08),
            (0.25, 0.75, 0.08),
            (0.95, 0.05, 0.08),
            (0.85, 0.15, 0.08),
            (0.75, 0.25, 0.08),
            (0.35, 0.65, 0.06),
            (0.45, 0.55, 0.06),
            (0.55, 0.45, 0.06),
            (0.65, 0.35, 0.06),
        ]

        for tx, ty, radius in tower_positions:
            distance = ((norm_x - tx) ** 2 + (norm_y - ty) ** 2) ** 0.5
            if distance < radius:
                return False

        if norm_x < 0.08 and norm_y > 0.92:
            return False
        if norm_x > 0.92 and norm_y < 0.08:
            return False

        return True

    def detect(self, minimap):
        """Detect the tracked champion on the minimap."""
        if minimap is None or not self._current_champion:
            return None

        templates = self._templates.get(self._current_champion, [])
        if not templates:
            return None

        minimap_filtered = self._filter_red_borders(minimap)

        best_match = None
        best_confidence = 0.0
        best_location = None

        minimap_height, minimap_width = minimap.shape[:2]

        for template in templates:
            if template.shape[0] > minimap_height or template.shape[1] > minimap_width:
                continue

            result = cv2.matchTemplate(
                minimap_filtered, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence and max_val > self._match_threshold:
                template_h, template_w = template.shape[:2]
                center_x = max_loc[0] + template_w // 2
                center_y = max_loc[1] + template_h // 2

                if self._is_valid_champion_location(center_x, center_y, minimap_width, minimap_height):
                    best_confidence = max_val
                    best_location = max_loc
                    best_match = template

        current_time = time.time()

        if best_match is not None and best_location is not None:
            template_h, template_w = best_match.shape[:2]
            center_x = best_location[0] + template_w // 2
            center_y = best_location[1] + template_h // 2

            norm_x, norm_y = normalize_position(
                center_x, center_y, minimap_width, minimap_height)
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

        self._update_visibility_history(False)

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

    def _update_visibility_history(self, is_visible):
        """Track visibility over multiple frames for stability."""
        self._visibility_history.append(is_visible)
        if len(self._visibility_history) > self._history_size:
            self._visibility_history.pop(0)

    def just_disappeared(self):
        """Check if the jungler just transitioned from visible to invisible."""
        if len(self._visibility_history) < 3:
            return False
        recent = self._visibility_history[-3:]
        return recent[0] and not recent[-1]

    def just_appeared(self):
        """Check if the jungler just became visible."""
        if len(self._visibility_history) < 3:
            return False
        recent = self._visibility_history[-3:]
        return not recent[0] and recent[-1]

    def is_stable_visible(self):
        """Check if jungler has been consistently visible."""
        if len(self._visibility_history) < 3:
            return False
        return all(self._visibility_history[-3:])

    def is_stable_invisible(self):
        """Check if jungler has been consistently invisible."""
        if len(self._visibility_history) < 3:
            return False
        return not any(self._visibility_history[-3:])

    def get_last_known_position(self):
        """Get the last known position of the jungler."""
        if self._last_detection:
            return self._last_detection.position
        return None

    def get_last_known_zone(self):
        """Get the last known zone of the jungler."""
        if self._last_detection:
            return self._last_detection.zone
        return None

    def reset(self):
        """Reset detector state for new game."""
        self._last_detection = None
        self._visibility_history = []


champion_detector = ChampionDetector()
grid_detector = champion_detector
