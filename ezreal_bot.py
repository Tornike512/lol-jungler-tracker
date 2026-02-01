"""
Ezreal AI Bot - Complete Gameplay Using Trained Models
======================================================
Uses both trained models:
- ezreal_movement_model.pt: Predicts movement direction
- ezreal_ability_model.pt: Predicts ability usage (Q, W, E, R, etc.)

Usage:
    python ezreal_bot.py

Requirements:
    - League of Legends in BORDERLESS or WINDOWED mode
    - Play as Ezreal
    - Live Client API enabled
    - Both model files in tlol/ folder

Controls:
    - F12 = Stop the bot
    - Move mouse to top-left corner = Emergency stop
    - Ctrl+C = Force stop
"""

import os
import sys
import time
import random
import requests
import urllib3
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import pyautogui
import pydirectinput
import keyboard

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

MOVEMENT_MODEL_PATH = "tlol/ezreal_movement_model.pt"
ABILITY_MODEL_PATH = "tlol/ezreal_ability_model.pt"
API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
API_TIMEOUT = 1.0

# Screen settings
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_CENTER_X = SCREEN_WIDTH // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2

# Ability cooldowns (seconds)
ABILITY_COOLDOWNS = {
    'q': 5.5,
    'w': 12,
    'e': 25,
    'r': 120,
    'd': 300,
    'f': 180,
}

# Ability mana costs
ABILITY_MANA = {
    'q': 28,
    'w': 50,
    'e': 90,
    'r': 100,
}

# Global state
stop_flag = False
last_ability_time = {'q': 0, 'w': 0, 'e': 0, 'r': 0, 'd': 0, 'f': 0}
last_move_time = 0
last_attack_time = 0


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class MovementModel(nn.Module):
    """Predicts movement direction (x, z deltas)."""

    def __init__(self, input_dim: int, output_dim: int = 81):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class AbilityModel(nn.Module):
    """Predicts ability usage (Q, W, E, R, etc.)."""

    def __init__(self, input_dim: int, num_abilities: int = 20):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
        )
        self.ability_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_abilities)
        ])

    def forward(self, x):
        shared_features = self.shared(x)
        outputs = [head(shared_features) for head in self.ability_heads]
        return torch.cat(outputs, dim=1)


def decode_action(action_id: int) -> Tuple[int, int]:
    """Decode action ID to movement direction (x_delta, z_delta)."""
    x_delta = (action_id // 9) - 4
    z_delta = (action_id % 9) - 4
    return x_delta, z_delta


# =============================================================================
# MODEL PREDICTOR
# =============================================================================

class EzrealPredictor:
    """Loads and runs both trained models."""

    def __init__(self):
        print("[*] Loading models...")

        # Load movement model
        move_checkpoint = torch.load(
            MOVEMENT_MODEL_PATH, map_location='cpu', weights_only=False)
        self.move_input_dim = move_checkpoint['input_dim']
        self.move_model = MovementModel(self.move_input_dim)
        self.move_model.load_state_dict(move_checkpoint['model_state_dict'])
        self.move_model.eval()
        self.move_mean = move_checkpoint['obs_mean']
        self.move_std = move_checkpoint['obs_std']
        print(f"[+] Movement model loaded (input_dim={self.move_input_dim})")

        # Load ability model
        ability_checkpoint = torch.load(
            ABILITY_MODEL_PATH, map_location='cpu', weights_only=False)
        self.ability_input_dim = ability_checkpoint['input_dim']
        self.num_abilities = ability_checkpoint['num_abilities']
        self.ability_model = AbilityModel(
            self.ability_input_dim, self.num_abilities)
        self.ability_model.load_state_dict(
            ability_checkpoint['model_state_dict'])
        self.ability_model.eval()
        self.ability_mean = ability_checkpoint['obs_mean']
        self.ability_std = ability_checkpoint['obs_std']
        print(
            f"[+] Ability model loaded (input_dim={self.ability_input_dim}, abilities={self.num_abilities})")

    def build_observation(self, game_state: Dict[str, Any], input_dim: int) -> np.ndarray:
        """Build observation vector from game state."""
        obs = np.zeros(input_dim, dtype=np.float32)

        # Basic stats
        game_time = game_state.get('game_time', 0)
        level = game_state.get('level', 1)
        current_health = game_state.get('current_health', 600)
        max_health = game_state.get('max_health', 600)
        current_mana = game_state.get('current_mana', 350)
        max_mana = game_state.get('max_mana', 350)

        health_pct = current_health / max_health if max_health > 0 else 1.0
        mana_pct = current_mana / max_mana if max_mana > 0 else 1.0

        # Fill observation vector
        obs[0] = game_time / 100.0
        obs[1] = min(game_time / 1800, 1.0)
        obs[2] = game_state.get('x', 7500) / 15000
        obs[3] = game_state.get('y', 7500) / 15000
        obs[4] = 0.0
        obs[5] = 0.0
        obs[6] = health_pct
        obs[7] = current_health / 1000.0
        obs[8] = max_health / 1000.0
        obs[9] = mana_pct
        obs[10] = current_mana / 500.0
        obs[11] = max_mana / 500.0
        obs[12] = level / 18.0
        obs[13] = game_state.get('attack_damage', 60) / 200.0
        obs[14] = game_state.get('ability_power', 0) / 500.0
        obs[15] = game_state.get('armor', 30) / 200.0
        obs[16] = game_state.get('magic_resist', 30) / 200.0
        obs[17] = game_state.get('attack_speed', 0.625)
        obs[18] = game_state.get('move_speed', 325) / 500.0
        obs[19] = game_state.get('minions_killed', 0) / 200.0
        obs[20] = game_state.get('kills', 0) / 20.0
        obs[21] = game_state.get('deaths', 0) / 20.0
        obs[22] = game_state.get('assists', 0) / 20.0
        obs[23] = game_state.get('current_gold', 0) / 5000.0
        obs[24] = (game_state.get('kills', 0) + game_state.get('assists', 0)
                   ) / (game_state.get('deaths', 0) + 1)
        obs[25] = game_state.get('minions_killed', 0) / (game_time / 60 + 1)
        obs[26] = 1 if game_state.get('x', 7500) < 7500 else 0
        obs[27] = 1 if game_state.get('y', 7500) > 7500 else 0
        obs[28] = health_pct * mana_pct
        obs[29] = level * game_state.get('attack_damage', 60) / 1000.0

        # Fill rest with zeros
        for i in range(30, input_dim):
            obs[i] = 0.0

        return obs

    def predict(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict both movement and abilities."""
        # Build observations
        move_obs = self.build_observation(game_state, self.move_input_dim)
        ability_obs = self.build_observation(
            game_state, self.ability_input_dim)

        # Normalize
        move_obs = (move_obs - self.move_mean) / (self.move_std + 1e-8)
        ability_obs = (ability_obs - self.ability_mean) / \
            (self.ability_std + 1e-8)

        with torch.no_grad():
            # Movement prediction
            move_input = torch.FloatTensor(move_obs).unsqueeze(0)
            move_output = self.move_model(move_input)
            move_probs = torch.softmax(move_output, dim=1)
            move_action = torch.argmax(move_output, dim=1).item()
            move_confidence = move_probs[0, move_action].item()

            # Ability prediction
            ability_input = torch.FloatTensor(ability_obs).unsqueeze(0)
            ability_output = self.ability_model(ability_input)
            ability_probs = ability_output[0].numpy()

        x_delta, z_delta = decode_action(move_action)

        return {
            'movement': {
                'action_id': move_action,
                'x_delta': x_delta,
                'z_delta': z_delta,
                'confidence': move_confidence,
            },
            'abilities': ability_probs,
        }


# =============================================================================
# GAME API FUNCTIONS
# =============================================================================

def get_game_data() -> Dict[str, Any]:
    """Fetch live game data from Riot API."""
    try:
        response = requests.get(API_URL, verify=False, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def find_ezreal(game_data: Dict) -> Dict[str, Any]:
    """Find Ezreal in the player list."""
    for player in game_data.get('allPlayers', []):
        if player.get('championName', '').lower() == 'ezreal':
            return player
    return {}


def extract_game_state(game_data: Dict, ezreal_data: Dict) -> Dict[str, Any]:
    """Extract game state for the AI model."""
    stats = ezreal_data.get('championStats', {})
    scores = ezreal_data.get('scores', {})

    return {
        'game_time': game_data.get('gameData', {}).get('gameTime', 0),
        'level': ezreal_data.get('level', 1),
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
        'x': 7500,
        'y': 7500,
    }


# =============================================================================
# ACTION FUNCTIONS
# =============================================================================

def move_in_direction(x_delta: int, z_delta: int, scale: int = 300):
    """Move in the predicted direction by clicking on screen."""
    offset_x = x_delta * scale
    offset_y = z_delta * scale  # Positive z = down on screen (toward bot lane)

    click_x = SCREEN_CENTER_X + offset_x
    click_y = SCREEN_CENTER_Y + offset_y

    click_x = max(300, min(SCREEN_WIDTH - 300, click_x))
    click_y = max(300, min(SCREEN_HEIGHT - 300, click_y))

    direction = []
    if x_delta > 0:
        direction.append("RIGHT")
    elif x_delta < 0:
        direction.append("LEFT")
    if z_delta > 0:
        direction.append("DOWN")  # Positive z = toward bot lane
    elif z_delta < 0:
        direction.append("UP")    # Negative z = toward top lane
    dir_str = "+".join(direction) if direction else "STOP"

    print(
        f"    [MOVE] Screen ({click_x}, {click_y}) | Delta ({x_delta:+d}, {z_delta:+d}) | {dir_str}")
    pydirectinput.moveTo(click_x, click_y)
    pydirectinput.click(button='right')


def attack_move():
    """Attack-move at center of screen."""
    offset_x = random.randint(-100, 100)
    offset_y = random.randint(-100, 50)
    click_x = SCREEN_CENTER_X + offset_x
    click_y = SCREEN_CENTER_Y + offset_y

    pydirectinput.moveTo(click_x, click_y)
    pydirectinput.keyDown('a')
    time.sleep(0.03)
    pydirectinput.click(button='left')
    pydirectinput.keyUp('a')


def use_ability(key: str, mana_available: int = 1000) -> bool:
    """Use an ability if off cooldown and have mana."""
    global last_ability_time
    current_time = time.time()
    time_since_last = current_time - last_ability_time.get(key, 0)
    cooldown = ABILITY_COOLDOWNS.get(key, 0)
    mana_cost = ABILITY_MANA.get(key, 0)

    if time_since_last > cooldown and mana_available >= mana_cost:
        pydirectinput.press(key)
        last_ability_time[key] = current_time
        return True
    return False


def use_q() -> bool:
    """Use Ezreal Q (Mystic Shot)."""
    aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
    aim_y = SCREEN_CENTER_Y + random.randint(-200, -50)
    pydirectinput.moveTo(aim_x, aim_y)
    return use_ability('q')


def use_w() -> bool:
    """Use Ezreal W (Essence Flux)."""
    aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
    aim_y = SCREEN_CENTER_Y + random.randint(-200, -50)
    pydirectinput.moveTo(aim_x, aim_y)
    return use_ability('w')


def use_e(direction: str = 'back') -> bool:
    """Use Ezreal E (Arcane Shift)."""
    if direction == 'back':
        aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
        aim_y = SCREEN_CENTER_Y + random.randint(100, 250)
    else:
        aim_x = SCREEN_CENTER_X + random.randint(-50, 50)
        aim_y = SCREEN_CENTER_Y + random.randint(-250, -100)
    pydirectinput.moveTo(aim_x, aim_y)
    return use_ability('e')


# =============================================================================
# MAIN BOT LOOP
# =============================================================================

def on_f12_press(e):
    """Stop the bot when F12 is pressed."""
    global stop_flag
    stop_flag = True
    print("\n[!] F12 pressed - Stopping...")


def interpret_abilities(ability_probs: np.ndarray) -> Dict[str, bool]:
    """Interpret ability model outputs."""
    # Map ability columns to keys (this is approximate based on training data)
    abilities = {
        # Col -17
        'q': ability_probs[3] > 0.5 if len(ability_probs) > 3 else False,
        # Col -16
        'w': ability_probs[4] > 0.5 if len(ability_probs) > 4 else False,
        # Col -15
        'e': ability_probs[5] > 0.5 if len(ability_probs) > 5 else False,
        # Col -14
        'r': ability_probs[6] > 0.5 if len(ability_probs) > 6 else False,
    }
    return abilities


def main():
    """Main bot execution loop."""
    global stop_flag, last_move_time, last_attack_time

    print("=" * 60)
    print("EZREAL AI BOT - COMPLETE")
    print("=" * 60)
    print("\nThis bot uses TWO neural networks:")
    print("  1. Movement model - predicts where to walk")
    print("  2. Ability model - predicts when to use Q/W/E/R")
    print("\nRequirements:")
    print("  - League of Legends in BORDERLESS or WINDOWED mode")
    print("  - Play as Ezreal")
    print("  - Live Client API enabled")
    print("\nControls:")
    print("  - F12 = Stop the bot")
    print("  - Mouse to top-left corner = Emergency stop")
    print("=" * 60)

    # Register F12 hotkey
    keyboard.on_press_key('f12', on_f12_press)

    # Setup pyautogui and pydirectinput
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02
    pydirectinput.FAILSAFE = True
    pydirectinput.PAUSE = 0.02

    # Load models
    try:
        print("\n[*] Loading models...")
        predictor = EzrealPredictor()
        print("[+] Both models ready!")
    except Exception as e:
        print(f"[!] Error loading models: {e}")
        return

    print("\n" + "=" * 60)
    print("Waiting for game...")
    print("=" * 60)

    # Countdown
    print("\nStarting in:")
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
                print("Ezreal not found. Play as Ezreal!", end='\r')
                time.sleep(1)
                continue

            if not game_detected:
                print(f"\n[INFO] Game detected! Ezreal found.")
                print("[INFO] Starting AI control...\n")
                game_detected = True

            if ezreal.get('isDead', False):
                print("  >> Dead, waiting to respawn...", end='\r')
                time.sleep(1)
                continue

            # Extract game state
            state = extract_game_state(game_data, ezreal)
            game_time = state['game_time']

            # Get predictions from both models
            prediction = predictor.predict(state)
            movement = prediction['movement']
            abilities = interpret_abilities(prediction['abilities'])

            # Display status
            minutes = int(game_time // 60)
            seconds = int(game_time % 60)
            health_pct = state['current_health'] / state['max_health'] * 100
            mana_pct = state['current_mana'] / state['max_mana'] * 100

            print(f"[{minutes:02d}:{seconds:02d}] Lvl {state['level']} | "
                  f"HP {health_pct:.0f}% | MP {mana_pct:.0f}% | "
                  f"CS {state['minions_killed']}")
            print(
                f"  Movement: ({movement['x_delta']:+d}, {movement['z_delta']:+d}) conf={movement['confidence']:.1%}")
            print(
                f"  Abilities: Q={abilities.get('q', False)} W={abilities.get('w', False)} E={abilities.get('e', False)} R={abilities.get('r', False)}")

            current_time = time.time()
            health_pct_val = state['current_health'] / state['max_health']
            mana = state['current_mana']

            # LOW HEALTH - Escape
            if health_pct_val < 0.3:
                print("  >> LOW HP! Escaping...")
                if use_e('back'):
                    print("  >> Used E to escape!")
                if health_pct_val < 0.2:
                    use_ability('f', mana)
                move_in_direction(-movement['x_delta'], -
                                  movement['z_delta'], scale=200)

            # NORMAL GAMEPLAY
            else:
                time_since_move = current_time - last_move_time

                # Move based on model prediction
                if time_since_move > 0.5:
                    move_in_direction(movement['x_delta'], movement['z_delta'])
                    last_move_time = current_time

                # Use abilities based on model prediction
                if abilities.get('q') and mana > 50:
                    if use_q():
                        print("  >> Model used Q")

                if abilities.get('w') and mana > 100:
                    if use_w():
                        print("  >> Model used W")

                if abilities.get('e') and mana > 90:
                    if use_e('forward'):
                        print("  >> Model used E")

                # Attack pattern
                time_since_attack = current_time - last_attack_time
                if time_since_attack > 0.5:
                    attack_move()
                    last_attack_time = current_time

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        print("\n\n[!] Bot stopped.")


if __name__ == "__main__":
    main()
