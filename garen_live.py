"""
Garen AI Live Game Assistant
=============================
Connects to a live League game and controls Garen automatically.
Run this while playing Garen in Practice Tool.
"""
import requests
import urllib3
import time
import sys
import random
import pyautogui
import keyboard

from garen_predict import GarenPredictor
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

# Disable pyautogui fail-safe for smoother operation (move mouse to corner to emergency stop)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02  # Small delay between actions

API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"

# Screen settings for 1920x1080 resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2

# Minimap settings for 1920x1080 resolution with default HUD scale
# CORRECTED based on diagnostic tool (minimap_test_4_Full_Bottom_Right.png)
MINIMAP_X = 1585      # Left edge of actual minimap (1550 + 35 offset)
MINIMAP_Y = 715       # Top edge of actual minimap (650 + 65 offset)
MINIMAP_SIZE = 300    # Actual minimap square size (not the capture region)
MAP_SIZE = 15000      # LoL map size in game units

# Top lane waypoint path (blue side) - from base to enemy tower
# These are sequential waypoints to guide Garen through top lane
TOP_LANE_WAYPOINTS = [
    (1000, 13500),   # 0: Fountain spawn
    (1500, 13200),   # 1: Base exit
    (2200, 13200),   # 2: Near inner tower
    (3000, 12800),   # 3: Between towers
    (4200, 11800),   # 4: Near outer tower (lane start)
    (5500, 10800),   # 5: Safe farming spot
    (6500, 10200),   # 6: Pushed up
    (7500, 9500),    # 7: Near river/enemy tower
]

# Current waypoint tracking
current_waypoint_index = 0
last_waypoint_time = 0


def get_nearest_waypoint(pos_x, pos_y):
    """Find the nearest waypoint to current position."""
    min_dist = float('inf')
    nearest_idx = 0
    for i, (wx, wy) in enumerate(TOP_LANE_WAYPOINTS):
        dist = ((pos_x - wx)**2 + (pos_y - wy)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx, min_dist


# Game state tracking
last_ability_time = {'q': 0, 'w': 0, 'e': 0, 'r': 0}
last_move_time = 0
ABILITY_COOLDOWNS = {'q': 8, 'w': 20, 'e': 9,
                     'r': 120}  # Approximate cooldowns


def map_to_minimap(map_x, map_y):
    """Convert map coordinates to minimap screen coordinates."""
    map_x = max(0, min(MAP_SIZE, map_x))
    map_y = max(0, min(MAP_SIZE, map_y))
    screen_x = MINIMAP_X + (map_x / MAP_SIZE) * MINIMAP_SIZE
    screen_y = MINIMAP_Y + ((MAP_SIZE - map_y) / MAP_SIZE) * MINIMAP_SIZE
    return int(screen_x), int(screen_y)


def move_to_position(map_x, map_y):
    """Right-click on minimap to move to position."""
    screen_x, screen_y = map_to_minimap(map_x, map_y)
    print(
        f"  [DEBUG] move_to_position called: map=({map_x}, {map_y}) -> screen=({screen_x}, {screen_y})")
    print(f"  [DEBUG] Moving mouse to minimap ({screen_x}, {screen_y})")
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    print(f"  [DEBUG] Mouse now at {pyautogui.position()}")
    print(f"  [DEBUG] RIGHT-clicking at ({screen_x}, {screen_y})")
    pyautogui.click(screen_x, screen_y, button='right')
    print(f"  [DEBUG] Minimap click complete")


def attack_move_click():
    """Attack-move using 'A' key + left-click."""
    # Click near Garen (center of screen) - attack-move finds nearest enemy automatically
    # Click AHEAD of Garen to walk towards minions while attacking
    click_x = SCREEN_CENTER_X + random.randint(-150, 150)
    # Can click ahead or slightly behind
    click_y = SCREEN_CENTER_Y + random.randint(-100, 200)

    print(f"  [DEBUG] Attack-move: Moving mouse to ({click_x}, {click_y})")

    # First move mouse to position
    pyautogui.moveTo(click_x, click_y, duration=0.1)
    print(f"  [DEBUG] Mouse now at {pyautogui.position()}")

    # Use pyautogui hotkey for attack-move (more reliable than keyboard library for combo)
    print(f"  [DEBUG] Pressing 'a' key...")
    pyautogui.keyDown('a')
    time.sleep(0.05)
    print(f"  [DEBUG] Clicking LEFT at ({click_x}, {click_y})")
    pyautogui.click(click_x, click_y, button='left')
    print(f"  [DEBUG] Releasing 'a' key...")
    pyautogui.keyUp('a')
    print(f"  [DEBUG] Attack-move complete")
    time.sleep(0.1)


def right_click_attack():
    """Right-click ahead to move/attack."""
    click_x = SCREEN_CENTER_X + random.randint(-150, 150)
    click_y = SCREEN_CENTER_Y + random.randint(-200, -50)  # Move towards enemy
    print(f"  [DEBUG] Move/attack RIGHT click at ({click_x}, {click_y})")
    pyautogui.click(click_x, click_y, button='right')


def use_ability(key):
    """Press an ability key using pyautogui for better game compatibility."""
    current_time = time.time()
    time_since_last = current_time - last_ability_time.get(key, 0)
    cooldown = ABILITY_COOLDOWNS.get(key, 0)
    print(
        f"  [DEBUG] use_ability('{key}') called. Time since last: {time_since_last:.1f}s, cooldown: {cooldown}s")

    if time_since_last > cooldown:
        print(f"  [DEBUG] --> Ability '{key}' is READY, sending keypress")
        print(f"  [DEBUG] --> pyautogui.keyDown('{key}')")
        pyautogui.keyDown(key)
        time.sleep(0.05)
        print(f"  [DEBUG] --> pyautogui.keyUp('{key}')")
        pyautogui.keyUp(key)
        last_ability_time[key] = current_time
        print(f"  [DEBUG] --> Ability '{key}' used successfully!")
        return True
    else:
        print(
            f"  [DEBUG] --> Ability '{key}' on cooldown ({time_since_last:.1f}s / {cooldown}s)")
    return False


def farm_minions():
    """Farm minions by attack-moving."""
    # Always use attack-move for farming
    attack_move_click()


def get_lane_position(game_time, level):
    """Get appropriate lane position based on game time and level."""
    # Very early: walk to lane
    if game_time < 60:
        return TOP_LANE_POSITIONS[3]  # Walk to outer tower

    # Laning phase: position based on level (stay in lane to farm)
    # Higher level = can push further up
    if level <= 3:
        base_pos = TOP_LANE_POSITIONS[4]  # Safe farming spot
    elif level <= 6:
        base_pos = TOP_LANE_POSITIONS[5]  # Pushed up a bit
    else:
        base_pos = TOP_LANE_POSITIONS[5]  # Stay in lane area

    # Add randomness to not stand still
    return (base_pos[0] + random.randint(-300, 300), base_pos[1] + random.randint(-300, 300))


def get_game_data():
    """Fetch live game data from Riot API."""
    try:
        response = requests.get(API_URL, verify=False, timeout=1)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def find_garen(game_data):
    """Find Garen in the player list."""
    for player in game_data.get('allPlayers', []):
        if player.get('championName', '').lower() == 'garen':
            return player
    return None


def extract_game_state(game_data, garen_data, position):
    """Extract game state for the AI model."""
    stats = garen_data.get('championStats', {})
    scores = garen_data.get('scores', {})

    game_time = game_data.get('gameData', {}).get('gameTime', 0)
    level = garen_data.get('level', 1)

    # Position from screen capture
    est_x, est_y = position

    # Determine if winning (compare team gold/kills)
    my_team = garen_data.get('team', 'ORDER')
    team_kills = 0
    enemy_kills = 0
    for p in game_data.get('allPlayers', []):
        if p.get('team') == my_team:
            team_kills += p.get('scores', {}).get('kills', 0)
        else:
            enemy_kills += p.get('scores', {}).get('kills', 0)

    win = 1 if team_kills > enemy_kills else 0

    return {
        'x': est_x,
        'y': est_y,
        'level': level,
        'current_gold': scores.get('currentGold', 0),
        'total_gold': scores.get('currentGold', 0) * 2,  # Estimate
        'xp': level * 500,  # Estimate
        'minions_killed': scores.get('creepScore', 0),
        'jungle_minions': 0,
        'damage_done': stats.get('physicalDamageDealtToChampions', 0),
        'damage_taken': stats.get('physicalDamageTaken', 0),
        'game_time': game_time,
        'game_duration': 1800,  # Assume 30 min game
        'win': win
    }


def main():
    global stop_flag
    print("=" * 60)
    print("GAREN AI - AUTO PLAY MODE")
    print("=" * 60)
    print("\nStart a game as Garen (Practice Tool, Blue Side)")
    print("\nThe AI will:")
    print("  - DETECT position from minimap (screen capture)")
    print("  - Use TRAINED MODEL to decide where to go")
    print("  - Farm minions (attack-move)")
    print("  - Use abilities (E, Q)")
    print("\nControls:")
    print("  - F12 = Stop the script")
    print("  - Move mouse to top-left corner = Emergency stop")
    print("  - Ctrl+C = Stop the script")
    print("\nMake sure League is in BORDERLESS or WINDOWED mode!")
    print("\nIMPORTANT:")
    print("  - Uses A + LeftClick for attack-move (default keybind)")
    print("  - Make sure League window is FOCUSED when script runs!")
    print("  - Run this script as Administrator if inputs don't work")
    print("=" * 60)

    predictor = GarenPredictor()
    tracker = MinimapTracker()
    print("[+] Minimap tracker initialized")

    print("\n[TEST] Saving minimap debug image...")
    tracker.debug_capture("minimap_debug.png")
    print("[TEST] Check 'minimap_debug.png' to verify position detection!")
    print("[TEST] Green circle = detected position\n")

    print("Waiting for game...")
    print("(Keep League window focused!)\n")

    # 3 second countdown before starting
    print("Starting in:")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO!\n")

    last_prediction = None

    try:
        while not stop_flag:
            game_data = get_game_data()

            if not game_data:
                print("No game detected. Waiting...", end='\r')
                time.sleep(2)
                continue

            garen = find_garen(game_data)

            if not garen:
                print("Garen not found in game. Play as Garen!", end='\r')
                time.sleep(1)
                continue

            # First detection - give user time to switch to game
            if last_prediction is None:
                print(f"\n[INFO] Game detected! Garen found.")
                print("[INFO] Going to TOP LANE")
                print("[INFO] Starting in 3 seconds...\n")
                time.sleep(3)

            # Get position from screen capture
            pos_x, pos_y, minimap_pixel = tracker.get_player_position()

            # Extract state with actual position
            state = extract_game_state(game_data, garen, (pos_x, pos_y))

            # BLUE SIDE: Spawn is at bottom-left (1000, 13500)
            # Need to walk UP to top lane (y decreases as we go up on map)
            # Top lane is at y > 9000

            # Find nearest waypoint
            nearest_idx, dist_to_nearest = get_nearest_waypoint(pos_x, pos_y)
            print(
                f"  [DEBUG] Nearest waypoint: WP{nearest_idx}, dist={dist_to_nearest:.0f}")

            # Update waypoint index if we've reached current target
            if dist_to_nearest < 500:
                global current_waypoint_index
                current_waypoint_index = min(
                    nearest_idx + 1, len(TOP_LANE_WAYPOINTS) - 1)
                print(f"  [DEBUG] Advanced to WP{current_waypoint_index}")

            # Determine target based on game state
            level = state['level']
            game_time = state['game_time']

            if game_time < 60:
                # First minute: follow waypoints from spawn to lane
                target_idx = current_waypoint_index
            elif level <= 5:
                target_idx = 5  # Safe farming
            elif level <= 10:
                target_idx = 6  # Pushed up
            else:
                target_idx = 7  # Near enemy tower

            target_x, target_y = TOP_LANE_WAYPOINTS[target_idx]
            location = f"Top Lane WP{target_idx}"
            print(
                f"  [DEBUG] Target set to WP{target_idx}: ({target_x}, {target_y})")

            # Display
            game_time = state['game_time']
            minutes = int(game_time // 60)
            seconds = int(game_time % 60)

            detected = "OK" if minimap_pixel else "EST"
            print(f"[{minutes:02d}:{seconds:02d}] Lvl {state['level']} | CS {state['minions_killed']} | Pos: ({pos_x}, {pos_y}) [{detected}]")
            print(
                f"  >> Model says go to: {location} ({target_x}, {target_y})")

            # Game logic based on state
            if garen.get('isDead', False):
                print("  >> Dead, waiting to respawn...")
                time.sleep(1)
                continue

            # Calculate distance to target
            dist_to_target = ((pos_x - target_x)**2 +
                              (pos_y - target_y)**2)**0.5
            print(
                f"  [DEBUG] Distance to target WP{target_idx}: {dist_to_target:.0f} units")
            print(
                f"  [DEBUG] Current pos: ({pos_x}, {pos_y}), Target: ({target_x}, {target_y})")

            # Move to lane via minimap - more frequent clicks when far away
            current_time = time.time()
            global last_move_time
            time_since_last_move = current_time - last_move_time
            print(
                f"  [DEBUG] Time since last move: {time_since_last_move:.1f}s")

            # Move more frequently when far from target
            move_delay = 2.0 if dist_to_target > 3000 else 5.0
            print(f"  [DEBUG] Move delay threshold: {move_delay}s")

            if time_since_last_move > move_delay:
                print(
                    f"  [DEBUG] Move timer expired, checking if we should move...")
                if dist_to_target > 500:  # Move if more than 500 units away
                    print(
                        f"  [DEBUG] YES - Moving to WP{target_idx} (dist: {dist_to_target:.0f} > 500)")
                    move_to_position(target_x, target_y)
                    last_move_time = current_time
                else:
                    print(
                        f"  [DEBUG] NO - Already close enough (dist: {dist_to_target:.0f} <= 500)")
                    last_move_time = current_time
            else:
                print(
                    f"  [DEBUG] Move timer not expired yet ({time_since_last_move:.1f}s < {move_delay}s)")

            # Farm minions - attack-move multiple times to ensure minion aggro
            # Farm if we're close enough to target (in lane)
            in_top_lane = pos_y > 8000 and 1000 < pos_x < 9000
            should_farm = dist_to_target < 1500 or in_top_lane
            print(
                f"  [DEBUG] Farming check: in_top_lane={in_top_lane}, should_farm={should_farm}")

            if should_farm:
                if dist_to_target < 1500:
                    print(
                        f"  >> Farming at target (dist: {dist_to_target:.0f})")
                else:
                    print(f"  >> Farming in top lane area")
                for i in range(4):  # Multiple attack-moves per cycle
                    print(f"  [DEBUG] Farming iteration {i+1}/4")
                    farm_minions()
                    time.sleep(0.15)

                # Use E (spin) for wave clear - use it often when in lane
                rand_e = random.random()
                print(f"  [DEBUG] E chance: {rand_e:.2f} (need < 0.7)")
                if rand_e < 0.7:  # 70% chance each tick
                    if use_ability('e'):
                        print("  >> Used E (Spin)")
                        time.sleep(0.4)  # Let E channel for a bit

                # Use Q for extra damage / movement
                rand_q = random.random()
                print(f"  [DEBUG] Q chance: {rand_q:.2f} (need < 0.4)")
                if rand_q < 0.4:  # 40% chance
                    if use_ability('q'):
                        print("  >> Used Q")
            else:
                print(
                    f"  >> Too far from target (dist: {dist_to_target:.0f}), skipping farm...")

            last_prediction = {'x': target_x, 'y': target_y}
            time.sleep(0.2)  # Faster update rate

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
