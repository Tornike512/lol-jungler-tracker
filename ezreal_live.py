"""
Ezreal AI Live Game Assistant
==============================
Connects to a live League game and controls Ezreal automatically.
Uses the trained behavioral cloning model for movement decisions.
"""
import requests
import urllib3
import time
import sys
import random
import pyautogui
import keyboard

from ezreal_predict import EzrealPredictor
from minimap_tracker import MinimapTracker

# Global flag for stopping
stop_flag = False


def on_f12_press(e):
    """Stop the script when F12 is pressed."""
    global stop_flag
    stop_flag = True
    print("\n[!] F12 pressed - Stopping...")


# Register F12 hotkey
keyboard.on_press_key('f12', on_f12_press)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable pyautogui fail-safe for smoother operation
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"

# Screen settings for 1920x1080 resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2

# Minimap settings
MINIMAP_X = 1585
MINIMAP_Y = 715
MINIMAP_SIZE = 300
MAP_SIZE = 15000

# Bot lane waypoints (blue side ADC)
# Coordinate system: X increases going right, Y increases going UP
# Blue fountain is at ~(1350, 250), bot lane runs along BOTTOM of map
BOT_LANE_WAYPOINTS = [
    (1350, 400),     # 0: Fountain/spawn area
    (2200, 600),     # 1: Base exit
    (3500, 800),     # 2: Near inner tower
    (5000, 1000),    # 3: Between towers
    (6500, 1200),    # 4: Near outer tower (lane start)
    (8000, 1400),    # 5: Safe farming spot (middle of lane)
    (9500, 1600),    # 6: Pushed up
    (11000, 1800),   # 7: Near enemy tower
]

current_waypoint_index = 0

# Ezreal ability cooldowns (approximate at rank 1)
ABILITY_COOLDOWNS = {
    'q': 5.5,   # Mystic Shot
    'w': 12,    # Essence Flux
    'e': 25,    # Arcane Shift
    'r': 120,   # Trueshot Barrage
    'd': 300,   # Flash (summoner spell)
    'f': 180,   # Heal/Barrier (summoner spell)
}

# Ability mana costs
ABILITY_MANA = {
    'q': 28,
    'w': 50,
    'e': 90,
    'r': 100,
}

# Track ability usage
last_ability_time = {'q': 0, 'w': 0, 'e': 0, 'r': 0, 'd': 0, 'f': 0}
last_move_time = 0
last_attack_time = 0


def map_to_minimap(map_x, map_y):
    """Convert map coordinates to minimap screen coordinates."""
    map_x = max(0, min(MAP_SIZE, map_x))
    map_y = max(0, min(MAP_SIZE, map_y))
    screen_x = MINIMAP_X + (map_x / MAP_SIZE) * MINIMAP_SIZE
    screen_y = MINIMAP_Y + ((MAP_SIZE - map_y) / MAP_SIZE) * MINIMAP_SIZE
    return int(screen_x), int(screen_y)


def map_to_screen(map_x, map_y, player_x, player_y):
    """
    Convert map coordinates to screen coordinates relative to player.
    Assumes camera is centered on player (locked camera or spacebar held).
    """
    # Approximate: 1 game unit = ~0.07 screen pixels at default zoom
    pixels_per_unit = 0.07

    dx = (map_x - player_x) * pixels_per_unit
    dy = -(map_y - player_y) * pixels_per_unit  # Y is inverted

    screen_x = SCREEN_CENTER_X + dx
    screen_y = SCREEN_CENTER_Y + dy

    # Clamp to screen bounds (leave margin for UI)
    screen_x = max(100, min(SCREEN_WIDTH - 100, screen_x))
    screen_y = max(100, min(SCREEN_HEIGHT - 200, screen_y))

    return int(screen_x), int(screen_y)


def move_to_position_minimap(map_x, map_y):
    """Right-click on minimap to move to position."""
    screen_x, screen_y = map_to_minimap(map_x, map_y)
    print(f"  [CLICK] Minimap click at screen ({screen_x}, {screen_y}) for map ({map_x}, {map_y})")
    pyautogui.click(screen_x, screen_y, button='right')


def move_to_position_screen(target_x, target_y, player_x, player_y):
    """Right-click on game screen to move (better for kiting)."""
    screen_x, screen_y = map_to_screen(target_x, target_y, player_x, player_y)
    pyautogui.click(screen_x, screen_y, button='right')


def kite_move(direction='back'):
    """
    Quick kite movement for ADC playstyle.
    Ezreal should constantly move between auto attacks.
    """
    if direction == 'back':
        offset_y = random.randint(50, 150)
        offset_x = random.randint(-50, 50)
    elif direction == 'forward':
        offset_y = random.randint(-150, -50)
        offset_x = random.randint(-50, 50)
    else:  # side
        offset_y = random.randint(-30, 30)
        offset_x = random.randint(-150, 150)

    click_x = SCREEN_CENTER_X + offset_x
    click_y = SCREEN_CENTER_Y + offset_y

    pyautogui.click(click_x, click_y, button='right')


def attack_move_click():
    """Attack-move using 'A' key + left-click."""
    click_x = SCREEN_CENTER_X + random.randint(-100, 100)
    click_y = SCREEN_CENTER_Y + random.randint(-100, 50)

    pyautogui.moveTo(click_x, click_y, duration=0.05)
    pyautogui.keyDown('a')
    time.sleep(0.03)
    pyautogui.click(click_x, click_y, button='left')
    pyautogui.keyUp('a')


def use_ability(key, mana_available=1000):
    """Use an ability if off cooldown and have mana."""
    current_time = time.time()
    time_since_last = current_time - last_ability_time.get(key, 0)
    cooldown = ABILITY_COOLDOWNS.get(key, 0)
    mana_cost = ABILITY_MANA.get(key, 0)

    if time_since_last > cooldown and mana_available >= mana_cost:
        pyautogui.press(key)
        last_ability_time[key] = current_time
        return True
    return False


def use_q_skillshot(target_direction='forward'):
    """
    Use Ezreal Q (Mystic Shot) - line skillshot.
    Aims towards cursor/enemy direction.
    """
    if target_direction == 'forward':
        aim_x = SCREEN_CENTER_X + random.randint(-30, 30)
        aim_y = SCREEN_CENTER_Y + random.randint(-200, -100)
    else:
        aim_x = SCREEN_CENTER_X + random.randint(-100, 100)
        aim_y = SCREEN_CENTER_Y + random.randint(-150, 0)

    pyautogui.moveTo(aim_x, aim_y, duration=0.05)

    if use_ability('q'):
        return True
    return False


def use_w_skillshot():
    """Use Ezreal W (Essence Flux) - mark skillshot."""
    aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
    aim_y = SCREEN_CENTER_Y + random.randint(-200, -80)

    pyautogui.moveTo(aim_x, aim_y, duration=0.05)

    if use_ability('w'):
        return True
    return False


def use_e_blink(direction='back'):
    """
    Use Ezreal E (Arcane Shift) - blink ability.
    Use for escaping or repositioning.
    """
    if direction == 'back':
        aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
        aim_y = SCREEN_CENTER_Y + random.randint(100, 200)
    elif direction == 'forward':
        aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
        aim_y = SCREEN_CENTER_Y + random.randint(-200, -100)
    else:  # side
        aim_x = SCREEN_CENTER_X + random.randint(-200, 200)
        aim_y = SCREEN_CENTER_Y + random.randint(-50, 50)

    pyautogui.moveTo(aim_x, aim_y, duration=0.05)

    if use_ability('e'):
        return True
    return False


def get_game_data():
    """Fetch live game data from Riot API."""
    try:
        response = requests.get(API_URL, verify=False, timeout=1)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def find_ezreal(game_data):
    """Find Ezreal in the player list."""
    for player in game_data.get('allPlayers', []):
        if player.get('championName', '').lower() == 'ezreal':
            return player
    return None


def extract_game_state(game_data, ezreal_data, position):
    """Extract game state for the AI model."""
    stats = ezreal_data.get('championStats', {})
    scores = ezreal_data.get('scores', {})

    game_time = game_data.get('gameData', {}).get('gameTime', 0)
    level = ezreal_data.get('level', 1)

    pos_x, pos_y = position

    return {
        'x': pos_x,
        'y': pos_y,
        'game_time': game_time,
        'level': level,
        'current_health': stats.get('currentHealth', 600),
        'max_health': stats.get('maxHealth', 600),
        'current_mana': stats.get('resourceValue', 350),
        'max_mana': stats.get('resourceMax', 350),
        'attack_damage': stats.get('attackDamage', 60),
        'ability_power': stats.get('abilityPower', 0),
        'armor': stats.get('armor', 30),
        'magic_resist': stats.get('magicResist', 30),
        'attack_speed': stats.get('attackSpeed', 0.625),
        'move_speed': stats.get('moveSpeed', 325),
        'minions_killed': scores.get('creepScore', 0),
        'kills': scores.get('kills', 0),
        'deaths': scores.get('deaths', 0),
        'assists': scores.get('assists', 0),
        'current_gold': scores.get('currentGold', 0),
    }


def get_nearest_waypoint(pos_x, pos_y):
    """Find the nearest waypoint to current position."""
    min_dist = float('inf')
    nearest_idx = 0
    for i, (wx, wy) in enumerate(BOT_LANE_WAYPOINTS):
        dist = ((pos_x - wx)**2 + (pos_y - wy)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx, min_dist


def get_lane_target(pos_x, pos_y, game_time, level):
    """Get the appropriate lane position to move to."""
    global current_waypoint_index

    # Find nearest waypoint
    nearest_idx, dist = get_nearest_waypoint(pos_x, pos_y)

    # Advance waypoint if close enough
    if dist < 600:
        current_waypoint_index = min(nearest_idx + 1, len(BOT_LANE_WAYPOINTS) - 1)

    # If still near spawn, always walk to lane first
    if pos_x < 4000:
        # Walk through waypoints sequentially
        target_idx = max(current_waypoint_index, 4)  # At least go to lane start
    # Early game: follow waypoints to lane
    elif game_time < 120:
        target_idx = min(current_waypoint_index, 5)  # Go to farming spot
    elif level <= 5:
        target_idx = 5  # Safe farming spot
    elif level <= 10:
        target_idx = 6  # Pushed up
    else:
        target_idx = 6  # Stay in lane area

    return BOT_LANE_WAYPOINTS[target_idx], target_idx


def main():
    global stop_flag, last_move_time, last_attack_time

    print("=" * 60)
    print("EZREAL AI - AUTO PLAY MODE")
    print("=" * 60)
    print("\nStart a game as Ezreal (Practice Tool recommended)")
    print("\nThe AI will:")
    print("  - Use TRAINED MODEL to predict movement")
    print("  - Kite and attack-move (ADC playstyle)")
    print("  - Use Q frequently for poke/farming")
    print("  - Use E to escape when low HP")
    print("\nControls:")
    print("  - F12 = Stop the script")
    print("  - Move mouse to top-left corner = Emergency stop")
    print("  - Ctrl+C = Stop the script")
    print("\nMake sure League is in BORDERLESS or WINDOWED mode!")
    print("=" * 60)

    # Load model and tracker
    predictor = EzrealPredictor()
    tracker = MinimapTracker()
    print("[+] Minimap tracker initialized")

    print("\n[TEST] Saving minimap debug image...")
    tracker.debug_capture("minimap_debug_ezreal.png")
    print("[TEST] Check 'minimap_debug_ezreal.png' to verify position detection!\n")

    print("Waiting for game...")
    print("(Keep League window focused!)\n")

    # Countdown
    print("Starting in:")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO!\n")

    game_detected = False

    try:
        while not stop_flag:
            game_data = get_game_data()

            if not game_data:
                print("No game detected. Waiting...", end='\r')
                time.sleep(2)
                continue

            ezreal = find_ezreal(game_data)

            if not ezreal:
                print("Ezreal not found in game. Play as Ezreal!", end='\r')
                time.sleep(1)
                continue

            # First detection
            if not game_detected:
                print(f"\n[INFO] Game detected! Ezreal found.")
                print("[INFO] Starting AI control in 3 seconds...\n")
                time.sleep(3)
                game_detected = True

            # Check if dead
            if ezreal.get('isDead', False):
                print("  >> Dead, waiting to respawn...", end='\r')
                time.sleep(1)
                continue

            # Get position from minimap
            pos_x, pos_y, minimap_pixel = tracker.get_player_position()

            # Extract game state
            state = extract_game_state(game_data, ezreal, (pos_x, pos_y))
            game_time = state['game_time']

            # Get lane target for navigation
            lane_target, waypoint_idx = get_lane_target(pos_x, pos_y, game_time, state['level'])
            dist_to_lane = ((pos_x - lane_target[0])**2 + (pos_y - lane_target[1])**2)**0.5

            # Get model prediction
            target_x, target_y, prediction = predictor.get_move_target(
                pos_x, pos_y, state, scale=300
            )

            # Display status
            minutes = int(game_time // 60)
            seconds = int(game_time % 60)
            health_pct = state['current_health'] / state['max_health'] * 100
            mana_pct = state['current_mana'] / state['max_mana'] * 100

            detected = "OK" if minimap_pixel else "EST"
            src = prediction.get('source', 'model')[:3].upper()
            print(f"[{minutes:02d}:{seconds:02d}] Lvl {state['level']} | "
                  f"HP {health_pct:.0f}% | MP {mana_pct:.0f}% | "
                  f"CS {state['minions_killed']} | "
                  f"Pos: ({pos_x}, {pos_y}) [{detected}]")
            print(f"  >> [{src}] delta=({prediction['x_delta']:+d}, {prediction['z_delta']:+d}) | "
                  f"Lane WP{waypoint_idx} dist={dist_to_lane:.0f}")

            current_time = time.time()

            # Decision making based on health
            health_pct_val = state['current_health'] / state['max_health']
            mana = state['current_mana']

            # LOW HEALTH - Escape mode
            if health_pct_val < 0.3:
                print("  >> LOW HP! Escaping...")

                # Use E to escape if available
                if use_e_blink('back'):
                    print("  >> Used E (Arcane Shift) to escape!")
                    time.sleep(0.3)

                # Use summoner heal/barrier
                if health_pct_val < 0.2:
                    use_ability('f', mana)  # Heal/Barrier

                # Kite backwards
                for _ in range(3):
                    kite_move('back')
                    time.sleep(0.1)

            # WALKING TO LANE - Not in lane yet
            elif dist_to_lane > 1000:
                time_since_move = current_time - last_move_time

                # Click more frequently when far from lane
                click_delay = 1.0 if dist_to_lane > 3000 else 2.0

                if time_since_move > click_delay:
                    print(f"  >> Walking to lane (WP{waypoint_idx})...")
                    move_to_position_minimap(lane_target[0], lane_target[1])
                    last_move_time = current_time

            # IN LANE - ADC kiting pattern
            else:
                # Move based on model/heuristic prediction
                time_since_move = current_time - last_move_time

                if time_since_move > 0.3:
                    # Use predicted direction for movement
                    if abs(prediction['x_delta']) > 0 or abs(prediction['z_delta']) > 0:
                        move_to_position_screen(target_x, target_y, pos_x, pos_y)
                        last_move_time = current_time

                # Attack pattern
                time_since_attack = current_time - last_attack_time

                if time_since_attack > 0.4:
                    # Use Q frequently for poke/farm (low cooldown)
                    if random.random() < 0.6 and mana > 50:
                        if use_q_skillshot('forward'):
                            print("  >> Used Q (Mystic Shot)")

                    # Attack-move for auto attacks
                    attack_move_click()
                    last_attack_time = current_time

                    # Kite after attacking
                    time.sleep(0.1)
                    kite_move('back' if random.random() < 0.7 else 'side')

                # Use W occasionally for bonus damage
                if random.random() < 0.15 and mana > 100:
                    if use_w_skillshot():
                        print("  >> Used W (Essence Flux)")

            time.sleep(0.15)

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
