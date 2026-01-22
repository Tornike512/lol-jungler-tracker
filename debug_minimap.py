#!/usr/bin/env python3
"""
Debug script to visualize what the minimap detector is capturing.
Run this while in a League game to see if the minimap region is correct.
"""

import mss
import cv2
import numpy as np
from pathlib import Path
import requests
import urllib3
import io
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration - same as main script
MINIMAP_RATIO_WIDTH = 0.1425
MINIMAP_RATIO_HEIGHT = 0.253

API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
DDRAGON_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{champion}.png"

OUTPUT_DIR = Path(__file__).parent / "debug_output"


def get_game_data():
    try:
        response = requests.get(API_URL, timeout=1.0, verify=False)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_enemy_jungler_name():
    game_data = get_game_data()
    if not game_data:
        return None

    active_player_name = game_data.get("activePlayer", {}).get("summonerName")
    my_team = None
    for player in game_data.get("allPlayers", []):
        if player.get("summonerName") == active_player_name:
            my_team = player.get("team")
            break

    if not my_team:
        return None

    for player in game_data.get("allPlayers", []):
        if player.get("team") != my_team:
            if player.get("position", "").upper() == "JUNGLE":
                return player.get("championName")

    return None


def get_champion_icon(champion_name):
    """Download champion icon from Data Dragon."""
    try:
        # Get version
        response = requests.get(DDRAGON_VERSION_URL, timeout=5.0)
        version = response.json()[0]

        # Special name mappings
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

        url_name = name_mappings.get(champion_name, champion_name.replace(" ", "").replace("'", ""))
        url = DDRAGON_CHAMPION_URL.format(version=version, champion=url_name)

        response = requests.get(url, timeout=5.0)
        if response.status_code == 200:
            pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error downloading icon: {e}")
    return None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Minimap Debug Tool")
    print("=" * 60)

    # Get screen info
    sct = mss.mss()
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    screen_height = monitor["height"]

    print(f"\nScreen resolution: {screen_width}x{screen_height}")

    # Calculate minimap region
    minimap_width = int(screen_width * MINIMAP_RATIO_WIDTH)
    minimap_height = int(screen_height * MINIMAP_RATIO_HEIGHT)
    minimap_left = screen_width - minimap_width
    minimap_top = screen_height - minimap_height

    print(f"Minimap region: x={minimap_left}, y={minimap_top}, w={minimap_width}, h={minimap_height}")

    # Capture minimap
    region = {
        "left": minimap_left,
        "top": minimap_top,
        "width": minimap_width,
        "height": minimap_height
    }

    screenshot = sct.grab(region)
    minimap = np.array(screenshot)
    minimap = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)

    # Save minimap screenshot
    minimap_path = OUTPUT_DIR / "minimap_capture.png"
    cv2.imwrite(str(minimap_path), minimap)
    print(f"\nSaved minimap capture to: {minimap_path}")

    # Also capture full screen for reference
    full_screen = sct.grab(sct.monitors[1])
    full_img = np.array(full_screen)
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

    # Draw rectangle where we think minimap is
    cv2.rectangle(full_img, (minimap_left, minimap_top),
                  (minimap_left + minimap_width, minimap_top + minimap_height),
                  (0, 255, 0), 3)

    full_path = OUTPUT_DIR / "full_screen_with_region.png"
    cv2.imwrite(str(full_path), full_img)
    print(f"Saved full screen with minimap region marked: {full_path}")

    # Try to get enemy jungler and test matching
    jungler_name = get_enemy_jungler_name()

    if jungler_name:
        print(f"\nEnemy jungler: {jungler_name}")

        icon = get_champion_icon(jungler_name)
        if icon is not None:
            # Save the icon
            icon_path = OUTPUT_DIR / f"{jungler_name}_icon.png"
            cv2.imwrite(str(icon_path), icon)
            print(f"Saved champion icon: {icon_path}")

            # Test template matching at different scales
            icon_size_ratio = 0.12
            base_size = int(min(minimap_width, minimap_height) * icon_size_ratio)

            print(f"\nTemplate matching results (base icon size: {base_size}px):")
            print("-" * 40)

            best_match = 0
            best_scale = 0
            best_loc = (0, 0)

            for scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                size = int(base_size * scale)
                if size < 5:
                    continue

                template = cv2.resize(icon, (size, size))

                if template.shape[0] > minimap.shape[0] or template.shape[1] > minimap.shape[1]:
                    continue

                result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                print(f"  Scale {scale:.1f} ({size}px): confidence = {max_val:.4f}")

                if max_val > best_match:
                    best_match = max_val
                    best_scale = scale
                    best_loc = max_loc

            print("-" * 40)
            print(f"Best match: scale={best_scale:.1f}, confidence={best_match:.4f}")
            print(f"Location: {best_loc}")

            # Draw best match on minimap
            if best_match > 0:
                size = int(base_size * best_scale)
                minimap_marked = minimap.copy()
                cv2.rectangle(minimap_marked, best_loc,
                              (best_loc[0] + size, best_loc[1] + size),
                              (0, 255, 0), 2)
                marked_path = OUTPUT_DIR / "minimap_with_match.png"
                cv2.imwrite(str(marked_path), minimap_marked)
                print(f"\nSaved minimap with best match marked: {marked_path}")
    else:
        print("\nNo enemy jungler found. Make sure you're in a game with an enemy jungler.")

    print(f"\n{'=' * 60}")
    print(f"Check the images in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
