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
MINIMAP_X = 1647      # Left edge of minimap
MINIMAP_Y = 817       # Top edge of minimap
MINIMAP_SIZE = 267    # Width/height of minimap
MAP_SIZE = 15000      # LoL map size in game units

# Top lane positions (blue side) - along the top lane path
TOP_LANE_POSITIONS = [
    (1000, 13500),   # At fountain/base
    (2200, 13200),   # Near inner tower
    (3000, 12800),   # Between towers
    (4200, 11800),   # Near outer tower
    (5500, 10800),   # In lane (farming spot)
    (6500, 10200),   # Pushed up
    (7500, 9500),    # Near river
]

# Game state tracking
last_ability_time = {'q': 0, 'w': 0, 'e': 0, 'r': 0}
last_move_time = 0
ABILITY_COOLDOWNS = {'q': 8, 'w': 20, 'e': 9, 'r': 120}  # Approximate cooldowns


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
    pyautogui.click(screen_x, screen_y, button='right')


def attack_move_click():
    """Press A then left-click to attack-move (attacks nearest enemy)."""
    # Press A key to enter attack-move mode
    pyautogui.press('a')
    time.sleep(0.05)
    # Click near center-top of screen (towards enemy minions in top lane)
    click_x = SCREEN_CENTER_X + random.randint(-200, 200)
    click_y = SCREEN_CENTER_Y + random.randint(-200, 0)  # Upper half (towards enemy)
    pyautogui.click(click_x, click_y, button='left')


def right_click_attack():
    """Right-click ahead to move/attack."""
    click_x = SCREEN_CENTER_X + random.randint(-150, 150)
    click_y = SCREEN_CENTER_Y + random.randint(-200, -50)  # Move towards enemy
    pyautogui.click(click_x, click_y, button='right')


def use_ability(key):
    """Press an ability key."""
    current_time = time.time()
    if current_time - last_ability_time.get(key, 0) > ABILITY_COOLDOWNS.get(key, 0):
        pyautogui.press(key)
        last_ability_time[key] = current_time
        return True
    return False


def farm_minions():
    """Farm minions by attack-moving and right-clicking."""
    # Alternate between attack-move and right-click
    if random.random() < 0.5:
        attack_move_click()
    else:
        right_click_attack()


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


def extract_game_state(game_data, garen_data):
    """Extract game state for the AI model."""
    stats = garen_data.get('championStats', {})
    scores = garen_data.get('scores', {})

    # Position might be in different locations in the API response
    # Check activePlayer first (has position for local player)
    active_player = game_data.get('activePlayer', {})
    pos = active_player.get('position', {})

    # If position is not a dict, try to parse or use defaults
    if not isinstance(pos, dict):
        pos = {}

    game_time = game_data.get('gameData', {}).get('gameTime', 0)

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
        'x': pos.get('x', 7500),
        'y': pos.get('y', 7500),
        'level': garen_data.get('level', 1),
        'current_gold': scores.get('currentGold', 0),
        'total_gold': scores.get('currentGold', 0) * 2,  # Estimate
        'xp': garen_data.get('level', 1) * 500,  # Estimate
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
    print("  - Go to Top Lane")
    print("  - Farm minions (attack-move)")
    print("  - Use E (spin) for wave clear")
    print("  - Use Q for movement speed")
    print("\nControls:")
    print("  - F12 = Stop the script")
    print("  - Move mouse to top-left corner = Emergency stop")
    print("  - Ctrl+C = Stop the script")
    print("\nMake sure League is in BORDERLESS or WINDOWED mode!")
    print("=" * 60)

    predictor = GarenPredictor()

    print("\nWaiting for game...")
    print("(Switch to League window now!)\n")

    # 5 second countdown before starting
    print("Starting in:")
    for i in range(5, 0, -1):
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

            # Extract state
            state = extract_game_state(game_data, garen)
            prediction = {'x': 0, 'y': 0}  # Not using model predictions anymore

            # Display
            game_time = state['game_time']
            minutes = int(game_time // 60)
            seconds = int(game_time % 60)

            # Get lane position (always top lane)
            target_x, target_y = get_lane_position(game_time, state['level'])

            print(f"[{minutes:02d}:{seconds:02d}] Lvl {state['level']} | CS {state['minions_killed']} | Top Lane")

            # Game logic based on state
            if garen.get('isDead', False):
                print("  >> Dead, waiting to respawn...")
                time.sleep(1)
                continue

            # Move to lane via minimap every few seconds (not every tick)
            current_time = time.time()
            global last_move_time
            if current_time - last_move_time > 3:  # Move command every 3 seconds
                move_to_position(target_x, target_y)
                last_move_time = current_time

            # Farm minions aggressively
            farm_minions()
            time.sleep(0.1)
            farm_minions()

            # Use E (spin) for wave clear - use it often
            if random.random() < 0.4:  # 40% chance each tick
                if use_ability('e'):
                    print("  >> Used E (Spin)")

            # Use Q for extra damage / movement
            if random.random() < 0.2:  # 20% chance
                if use_ability('q'):
                    print("  >> Used Q")

            last_prediction = prediction
            time.sleep(0.2)  # Faster update rate

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
