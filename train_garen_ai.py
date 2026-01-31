"""
Garen AI Trainer - Behavioral Cloning
======================================
Trains a neural network to predict Garen's positioning and decisions
based on game state data extracted from pro player replays.
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try to import PyTorch, fall back to sklearn if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    USE_PYTORCH = True
except ImportError:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    USE_PYTORCH = False
    print("PyTorch not found, using sklearn instead")

DATA_FILE = "garen_training_data.csv"
MODEL_FILE = "garen_model.pt" if USE_PYTORCH else "garen_model.pkl"


class GarenNet(nn.Module):
    """Neural network for predicting Garen's next position."""

    def __init__(self, input_size, hidden_size=256):
        super(GarenNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: next x, y position
        )

    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data():
    """Load CSV and prepare features for training."""
    print("\n[1/4] Loading data...")

    if not os.path.exists(DATA_FILE):
        print(f"[!] Data file not found: {DATA_FILE}")
        print("    Run garen_pipeline.py first to collect data.")
        return None, None, None, None

    df = pd.read_csv(DATA_FILE)
    print(f"    Loaded {len(df)} rows")

    if len(df) < 100:
        print(f"[!] Not enough data for training (need at least 100 rows)")
        return None, None, None, None

    # Feature engineering
    print("\n[2/4] Engineering features...")

    # Calculate position changes (what we want to predict)
    df['next_x'] = df.groupby((df['timestamp'].diff() < 0).cumsum())['x'].shift(-1)
    df['next_y'] = df.groupby((df['timestamp'].diff() < 0).cumsum())['y'].shift(-1)

    # Remove rows without next position
    df = df.dropna(subset=['next_x', 'next_y'])

    # Normalize positions to 0-1 range (map is roughly 15000x15000)
    MAP_SIZE = 15000
    df['x_norm'] = df['x'] / MAP_SIZE
    df['y_norm'] = df['y'] / MAP_SIZE
    df['next_x_norm'] = df['next_x'] / MAP_SIZE
    df['next_y_norm'] = df['next_y'] / MAP_SIZE

    # Calculate game progress (0-1)
    df['game_progress'] = df['timestamp'] / df['game_duration']

    # Gold efficiency
    df['gold_per_min'] = df['total_gold'] / (df['timestamp'] / 60 + 1)

    # CS per minute
    df['cs_per_min'] = df['minions_killed'] / (df['timestamp'] / 60 + 1)

    # Add velocity features (how fast position is changing)
    df['prev_x'] = df.groupby((df['timestamp'].diff() < 0).cumsum())['x'].shift(1)
    df['prev_y'] = df.groupby((df['timestamp'].diff() < 0).cumsum())['y'].shift(1)
    df['velocity_x'] = (df['x'] - df['prev_x'].fillna(df['x'])) / MAP_SIZE
    df['velocity_y'] = (df['y'] - df['prev_y'].fillna(df['y'])) / MAP_SIZE

    # Add quadrant features (which part of map)
    df['in_blue_side'] = (df['x'] < 7500).astype(int)
    df['in_top_half'] = (df['y'] > 7500).astype(int)

    # Damage ratio (offensive vs defensive)
    df['damage_ratio'] = df['damage_done'] / (df['damage_taken'] + 1)

    # Features for the model
    feature_cols = [
        'x_norm', 'y_norm',  # Current position
        'velocity_x', 'velocity_y',  # Movement direction
        'level', 'current_gold', 'total_gold',
        'xp', 'minions_killed', 'jungle_minions',
        'damage_done', 'damage_taken', 'damage_ratio',
        'game_progress', 'gold_per_min', 'cs_per_min',
        'in_blue_side', 'in_top_half',
        'win'  # Include win as context
    ]

    target_cols = ['next_x_norm', 'next_y_norm']

    # Drop any remaining NaN
    df = df.dropna(subset=feature_cols + target_cols)

    print(f"    Final dataset: {len(df)} samples")
    print(f"    Features: {len(feature_cols)}")

    X = df[feature_cols].values
    y = df[target_cols].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    with open('garen_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test


def train_pytorch(X_train, X_test, y_train, y_test):
    """Train using PyTorch."""
    print("\n[3/4] Training PyTorch model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Using device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model
    model = GarenNet(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with learning rate scheduler
    epochs = 200
    best_loss = float('inf')
    patience = 20
    no_improve = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_FILE)
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    print(f"\n    Best validation loss: {best_loss:.6f}")
    return model


def train_sklearn(X_train, X_test, y_train, y_test):
    """Train using sklearn (fallback)."""
    print("\n[3/4] Training sklearn model...")

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\n    Train R2: {train_score:.4f}")
    print(f"    Test R2: {test_score:.4f}")

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("\n[4/4] Evaluating model...")

    if USE_PYTORCH:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            predictions = model(X_test_t).cpu().numpy()
    else:
        predictions = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))

    # Convert back to map coordinates for interpretability
    MAP_SIZE = 15000
    mae_coords = mae * MAP_SIZE

    print(f"    Mean Squared Error: {mse:.6f}")
    print(f"    Mean Absolute Error: {mae:.6f}")
    print(f"    Average position error: {mae_coords:.0f} units")
    print(f"    (Map is ~15000 units, so {mae_coords/150:.1f}% of map)")


def main():
    print("=" * 60)
    print("GAREN AI TRAINER")
    print("=" * 60)

    # Load data
    result = load_and_preprocess_data()
    if result[0] is None:
        return

    X_train, X_test, y_train, y_test = result

    # Train model
    if USE_PYTORCH:
        model = train_pytorch(X_train, X_test, y_train, y_test)
    else:
        model = train_sklearn(X_train, X_test, y_train, y_test)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Model saved to: {MODEL_FILE}")
    print(f"  Scaler saved to: garen_scaler.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
