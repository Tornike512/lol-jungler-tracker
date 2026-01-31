"""
Garen AI Position Predictor
============================
Uses the trained model to predict where Garen should position
based on current game state.
"""
import torch
import pickle
import numpy as np

MODEL_FILE = "garen_model.pt"
SCALER_FILE = "garen_scaler.pkl"
MAP_SIZE = 15000


class GarenNet(torch.nn.Module):
    """Same architecture as training."""

    def __init__(self, input_size=19, hidden_size=256):
        super(GarenNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


class GarenPredictor:
    def __init__(self):
        # Load model
        self.model = GarenNet(input_size=19)
        self.model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        self.model.eval()

        # Load scaler
        with open(SCALER_FILE, 'rb') as f:
            self.scaler = pickle.load(f)

        # Track previous position for velocity
        self.prev_x = None
        self.prev_y = None

        print("[+] Garen AI model loaded (v2 - 19 features)")

    def predict_position(self, game_state):
        """
        Predict next position based on game state.

        Args:
            game_state: dict with keys:
                - x, y: current position (map coordinates)
                - level: champion level (1-18)
                - current_gold: current gold
                - total_gold: total gold earned
                - xp: experience points
                - minions_killed: CS
                - jungle_minions: jungle CS
                - damage_done: total damage to champions
                - damage_taken: total damage taken
                - game_time: game time in seconds
                - game_duration: expected game duration
                - win: 1 if winning, 0 if losing (optional)

        Returns:
            dict with predicted x, y coordinates
        """
        # Current position
        x = game_state.get('x', 7500)
        y = game_state.get('y', 7500)
        x_norm = x / MAP_SIZE
        y_norm = y / MAP_SIZE

        # Velocity (movement direction)
        if self.prev_x is not None:
            velocity_x = (x - self.prev_x) / MAP_SIZE
            velocity_y = (y - self.prev_y) / MAP_SIZE
        else:
            velocity_x = 0
            velocity_y = 0
        self.prev_x = x
        self.prev_y = y

        # Basic stats
        level = game_state.get('level', 1)
        current_gold = game_state.get('current_gold', 0)
        total_gold = game_state.get('total_gold', 0)
        xp = game_state.get('xp', 0)
        minions_killed = game_state.get('minions_killed', 0)
        jungle_minions = game_state.get('jungle_minions', 0)
        damage_done = game_state.get('damage_done', 0)
        damage_taken = game_state.get('damage_taken', 0)
        game_time = game_state.get('game_time', 0)
        game_duration = game_state.get('game_duration', 1800)
        win = game_state.get('win', 0.5)

        # Derived features
        game_progress = game_time / game_duration if game_duration > 0 else 0
        gold_per_min = total_gold / (game_time / 60 + 1) if game_time > 0 else 0
        cs_per_min = minions_killed / (game_time / 60 + 1) if game_time > 0 else 0
        damage_ratio = damage_done / (damage_taken + 1)
        in_blue_side = 1 if x < 7500 else 0
        in_top_half = 1 if y > 7500 else 0

        # Feature vector (19 features, same order as training)
        features = np.array([[
            x_norm, y_norm,
            velocity_x, velocity_y,
            level, current_gold, total_gold,
            xp, minions_killed, jungle_minions,
            damage_done, damage_taken, damage_ratio,
            game_progress, gold_per_min, cs_per_min,
            in_blue_side, in_top_half,
            win
        ]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features_scaled)
            output = self.model(input_tensor).numpy()[0]

        # Convert back to map coordinates
        pred_x = output[0] * MAP_SIZE
        pred_y = output[1] * MAP_SIZE

        return {
            'x': int(pred_x),
            'y': int(pred_y),
            'description': self._describe_position(pred_x, pred_y)
        }

    def _describe_position(self, x, y):
        """Describe the position in game terms."""
        # Map regions (approximate)
        if x < 5000:
            if y < 5000:
                return "Blue side bot jungle"
            elif y > 10000:
                return "Blue side top jungle"
            else:
                return "Blue side mid jungle"
        elif x > 10000:
            if y < 5000:
                return "Red side bot jungle"
            elif y > 10000:
                return "Red side top jungle"
            else:
                return "Red side mid jungle"
        else:
            if y < 4000:
                return "Bot lane"
            elif y > 11000:
                return "Top lane"
            else:
                return "Mid lane area"


def demo():
    """Demo the predictor with sample game states."""
    predictor = GarenPredictor()

    print("\n" + "=" * 50)
    print("GAREN POSITION PREDICTOR DEMO")
    print("=" * 50)

    # Early game (laning)
    early_game = {
        'x': 2000, 'y': 12000,  # Top lane blue side
        'level': 3,
        'current_gold': 200,
        'total_gold': 800,
        'xp': 500,
        'minions_killed': 25,
        'jungle_minions': 0,
        'damage_done': 500,
        'damage_taken': 400,
        'game_time': 300,  # 5 minutes
        'game_duration': 1800,
        'win': 0.5
    }

    print("\n[Early Game - Level 3, 5 minutes]")
    print(f"  Current: Top lane blue side (2000, 12000)")
    result = predictor.predict_position(early_game)
    print(f"  Predicted: ({result['x']}, {result['y']}) - {result['description']}")

    # Mid game (roaming)
    mid_game = {
        'x': 7500, 'y': 7500,  # Mid area
        'level': 11,
        'current_gold': 500,
        'total_gold': 8000,
        'xp': 8000,
        'minions_killed': 150,
        'jungle_minions': 10,
        'damage_done': 8000,
        'damage_taken': 5000,
        'game_time': 1200,  # 20 minutes
        'game_duration': 1800,
        'win': 1  # Winning
    }

    print("\n[Mid Game - Level 11, 20 minutes, WINNING]")
    print(f"  Current: Mid area (7500, 7500)")
    result = predictor.predict_position(mid_game)
    print(f"  Predicted: ({result['x']}, {result['y']}) - {result['description']}")

    # Late game (teamfight)
    late_game = {
        'x': 10000, 'y': 8000,  # Near enemy base
        'level': 16,
        'current_gold': 1500,
        'total_gold': 15000,
        'xp': 15000,
        'minions_killed': 280,
        'jungle_minions': 20,
        'damage_done': 25000,
        'damage_taken': 18000,
        'game_time': 1800,  # 30 minutes
        'game_duration': 2000,
        'win': 0  # Losing
    }

    print("\n[Late Game - Level 16, 30 minutes, LOSING]")
    print(f"  Current: Near enemy base (10000, 8000)")
    result = predictor.predict_position(late_game)
    print(f"  Predicted: ({result['x']}, {result['y']}) - {result['description']}")


if __name__ == "__main__":
    demo()
