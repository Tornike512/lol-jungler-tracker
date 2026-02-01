"""
Ezreal AI Movement Predictor
=============================
Uses the trained behavioral cloning model to predict movement directions.
"""
import torch
import torch.nn as nn
import numpy as np

MODEL_FILE = "tlol/ezreal_model_best.pt"
MAP_SIZE = 15000


class EzrealModel(nn.Module):
    """Neural network for Ezreal behavioral cloning (must match training)."""

    def __init__(self, input_dim, output_dim=81, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def decode_action(action_id):
    """Decode action ID to movement direction."""
    x_delta = (action_id // 9) - 4  # -4 to 4
    z_delta = (action_id % 9) - 4   # -4 to 4
    return x_delta, z_delta


class EzrealPredictor:
    def __init__(self, model_path=MODEL_FILE):
        print("[*] Loading Ezreal model...")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        self.input_dim = checkpoint['input_dim']
        self.output_dim = checkpoint['output_dim']
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']

        # Create and load model
        self.model = EzrealModel(self.input_dim, self.output_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Track previous position for velocity
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None

        # Track prediction history for variety detection
        self.prediction_history = []
        self.use_heuristics = False

        print(f"[+] Ezreal model loaded (input_dim={self.input_dim}, output_dim={self.output_dim})")
        if 'val_accuracy' in checkpoint:
            print(f"[+] Model validation accuracy: {checkpoint['val_accuracy']:.2%}")

    def build_observation(self, game_state):
        """
        Build observation vector from game state.
        Maps Live Client API data to approximate the TLoL training features.

        The TLoL dataset has ~100 observation columns including:
        - Time features (game time, spawn timers)
        - Player state (health, mana, position, level)
        - Ability states (cooldowns, levels)
        - Combat stats
        """
        obs = np.zeros(self.input_dim, dtype=np.float32)

        # Extract values from game state
        x = game_state.get('x', 7500)
        y = game_state.get('y', 7500)
        game_time = game_state.get('game_time', 0)
        level = game_state.get('level', 1)

        # Health and mana
        current_health = game_state.get('current_health', 600)
        max_health = game_state.get('max_health', 600)
        current_mana = game_state.get('current_mana', 350)
        max_mana = game_state.get('max_mana', 350)
        health_pct = current_health / max_health if max_health > 0 else 1.0
        mana_pct = current_mana / max_mana if max_mana > 0 else 1.0

        # Stats
        ad = game_state.get('attack_damage', 60)
        ap = game_state.get('ability_power', 0)
        armor = game_state.get('armor', 30)
        mr = game_state.get('magic_resist', 30)
        attack_speed = game_state.get('attack_speed', 0.625)
        move_speed = game_state.get('move_speed', 325)

        # Combat
        cs = game_state.get('minions_killed', 0)
        kills = game_state.get('kills', 0)
        deaths = game_state.get('deaths', 0)
        assists = game_state.get('assists', 0)
        gold = game_state.get('current_gold', 0)

        # Velocity
        velocity_x = 0.0
        velocity_y = 0.0
        if self.prev_x is not None and self.prev_time is not None:
            dt = game_time - self.prev_time
            if dt > 0:
                velocity_x = (x - self.prev_x) / (dt * 100)
                velocity_y = (y - self.prev_y) / (dt * 100)

        self.prev_x = x
        self.prev_y = y
        self.prev_time = game_time

        # Normalize position to [0, 1]
        x_norm = x / MAP_SIZE
        y_norm = y / MAP_SIZE

        # Game progress
        game_progress = min(game_time / 1800, 1.0)  # Normalize to ~30 min game

        # Build observation vector
        # These indices are approximate mappings to TLoL features
        obs[0] = game_time / 100.0  # Time scaled
        obs[1] = game_progress
        obs[2] = x_norm
        obs[3] = y_norm
        obs[4] = velocity_x
        obs[5] = velocity_y
        obs[6] = health_pct
        obs[7] = current_health / 1000.0
        obs[8] = max_health / 1000.0
        obs[9] = mana_pct
        obs[10] = current_mana / 500.0
        obs[11] = max_mana / 500.0
        obs[12] = level / 18.0
        obs[13] = ad / 200.0
        obs[14] = ap / 500.0
        obs[15] = armor / 200.0
        obs[16] = mr / 200.0
        obs[17] = attack_speed
        obs[18] = move_speed / 500.0
        obs[19] = cs / 200.0
        obs[20] = kills / 20.0
        obs[21] = deaths / 20.0
        obs[22] = assists / 20.0
        obs[23] = gold / 5000.0

        # Fill remaining with derived features
        obs[24] = (kills + assists) / (deaths + 1)  # KDA ratio
        obs[25] = cs / (game_time / 60 + 1)  # CS per minute
        obs[26] = 1 if x < MAP_SIZE / 2 else 0  # Blue side
        obs[27] = 1 if y > MAP_SIZE / 2 else 0  # Top half
        obs[28] = health_pct * mana_pct  # Resource availability
        obs[29] = level * ad / 1000.0  # Power estimate

        # Enemy proximity estimates (unknown, use neutral values)
        for i in range(30, min(60, self.input_dim)):
            obs[i] = 0.5

        # Ability states (unknown exact cooldowns, fill with defaults)
        for i in range(60, self.input_dim):
            obs[i] = 0.0

        return obs

    def predict_movement(self, game_state):
        """
        Predict movement direction based on game state.

        Returns:
            dict with:
                - action_id: raw action class (0-80)
                - x_delta: movement in x direction (-4 to 4)
                - z_delta: movement in z direction (-4 to 4)
                - confidence: prediction confidence
                - top_actions: top 5 predicted actions
                - source: 'model' or 'heuristic'
        """
        # Build observation
        obs = self.build_observation(game_state)

        # Try model prediction first
        # Use simple standardization instead of training stats (distribution mismatch)
        obs_normalized = (obs - obs.mean()) / (obs.std() + 1e-8)

        with torch.no_grad():
            input_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            # Get top prediction
            action_id = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, action_id].item()

            # Get top 5
            top5_vals, top5_ids = torch.topk(probs, 5, dim=1)
            top_actions = [
                (idx.item(), decode_action(idx.item()), prob.item())
                for idx, prob in zip(top5_ids[0], top5_vals[0])
            ]

        # Track predictions to detect if model is stuck
        self.prediction_history.append(action_id)
        if len(self.prediction_history) > 20:
            self.prediction_history.pop(0)

        # Check if model is degenerate (same prediction too often)
        if len(self.prediction_history) >= 10:
            unique_preds = len(set(self.prediction_history[-10:]))
            if unique_preds <= 2:
                self.use_heuristics = True

        x_delta, z_delta = decode_action(action_id)
        source = 'model'

        # Use heuristics if model is stuck or confidence is suspiciously high
        if self.use_heuristics or confidence > 0.99:
            x_delta, z_delta = self._heuristic_movement(game_state)
            # Encode back to action_id for consistency
            action_id = (x_delta + 4) * 9 + (z_delta + 4)
            source = 'heuristic'

        return {
            'action_id': action_id,
            'x_delta': x_delta,
            'z_delta': z_delta,
            'confidence': confidence,
            'top_actions': top_actions,
            'source': source
        }

    def _heuristic_movement(self, game_state):
        """
        Heuristic movement when model fails.
        Uses game knowledge to decide movement.
        """
        import random

        x = game_state.get('x', 7500)
        y = game_state.get('y', 7500)
        health_pct = game_state.get('current_health', 600) / max(game_state.get('max_health', 600), 1)
        game_time = game_state.get('game_time', 0)

        # Bot lane target area (for blue side ADC)
        BOT_LANE_X = 10000
        BOT_LANE_Y = 3000

        # If low health, move towards base
        if health_pct < 0.3:
            # Move towards fountain (bottom-left for blue side)
            dx = -3 if x > 3000 else random.randint(-1, 1)
            dy = -3 if y > 3000 else random.randint(-1, 1)
            return dx, dy

        # If in fountain/base area, move to lane
        if x < 3000 and y < 3000:
            # Move towards bot lane (right and slightly up)
            dx = random.randint(2, 4)
            dy = random.randint(0, 2)
            return dx, dy

        # If not in bot lane yet (x < 8000), navigate there
        if x < 8000:
            # Walking to lane - move right with some variation
            dx = random.randint(2, 4)
            dy = random.randint(-1, 1)
            return dx, dy

        # In lane - kiting movements
        # Early game: stay closer to tower
        if game_time < 180:  # First 3 minutes
            # Small random movements, slightly towards tower
            dx = random.randint(-2, 1)
            dy = random.randint(-1, 1)
        else:
            # After laning starts, more aggressive positioning
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)

        return dx, dy

    def get_move_target(self, current_x, current_y, game_state, scale=200):
        """
        Get target position to move towards.

        Args:
            current_x, current_y: Current position in game coordinates
            game_state: Current game state dict
            scale: How far to move in game units per delta unit

        Returns:
            (target_x, target_y): Target position to click
        """
        prediction = self.predict_movement(game_state)

        # Convert delta to target position
        target_x = current_x + prediction['x_delta'] * scale
        target_y = current_y + prediction['z_delta'] * scale

        # Clamp to map bounds
        target_x = max(0, min(MAP_SIZE, target_x))
        target_y = max(0, min(MAP_SIZE, target_y))

        return target_x, target_y, prediction


def demo():
    """Demo the predictor with sample game states."""
    predictor = EzrealPredictor()

    print("\n" + "=" * 50)
    print("EZREAL MOVEMENT PREDICTOR DEMO")
    print("=" * 50)

    # Sample game state
    game_state = {
        'x': 5000, 'y': 10000,  # Bot lane position
        'game_time': 300,  # 5 minutes
        'level': 4,
        'current_health': 500,
        'max_health': 700,
        'current_mana': 200,
        'max_mana': 400,
        'attack_damage': 75,
        'ability_power': 10,
        'armor': 35,
        'magic_resist': 30,
        'attack_speed': 0.7,
        'move_speed': 325,
        'minions_killed': 35,
        'kills': 1,
        'deaths': 0,
        'assists': 2,
        'current_gold': 800,
    }

    print("\n[Test 1 - Laning Phase, Bot Lane]")
    print(f"  Position: ({game_state['x']}, {game_state['y']})")

    target_x, target_y, pred = predictor.get_move_target(
        game_state['x'], game_state['y'], game_state
    )

    print(f"  Prediction: action={pred['action_id']}, delta=({pred['x_delta']}, {pred['z_delta']})")
    print(f"  Confidence: {pred['confidence']:.2%}")
    print(f"  Move target: ({target_x:.0f}, {target_y:.0f})")
    print(f"  Top 5 actions:")
    for action_id, (dx, dz), prob in pred['top_actions']:
        print(f"    - Action {action_id}: ({dx:+d}, {dz:+d}) = {prob:.2%}")


if __name__ == "__main__":
    demo()
