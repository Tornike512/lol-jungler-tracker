"""
Train Ezreal Bot using Behavioral Cloning
==========================================
Trains a neural network to imitate Ezreal gameplay from the TLoL dataset.

The model learns to predict movement direction based on game state.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class EzrealDataset(Dataset):
    """PyTorch Dataset for Ezreal gameplay data (numpy format)."""

    def __init__(self, data_dir, max_games=None, obs_cols=100, action_type="movement"):
        """
        Args:
            data_dir: Path to the NP folder with .npy files
            max_games: Max number of games to load
            obs_cols: Number of observation columns to use (from start)
            action_type: "movement" (predict direction) or "ability" (predict ability usage)
        """
        self.data_dir = Path(data_dir)
        self.npy_files = list(self.data_dir.glob("*.npy"))
        self.action_type = action_type

        if max_games:
            self.npy_files = self.npy_files[:max_games]

        print(f"Loading {len(self.npy_files)} games...")

        # Load all games
        all_obs = []
        all_actions = []

        for npy_file in tqdm(self.npy_files, desc="Loading games"):
            try:
                data = np.load(npy_file, allow_pickle=True).astype(np.float32)

                # Replace inf/nan with finite values
                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

                # Extract observations (first N columns)
                obs = data[:, :obs_cols]

                # Extract actions based on type
                if action_type == "movement":
                    # Movement: columns -17 (x_delta) and -16 (z_delta)
                    # Range is approximately -4 to 4
                    x_delta = data[:, -17].astype(np.int32)
                    z_delta = data[:, -16].astype(np.int32)

                    # Encode as single class: (x + 4) * 9 + (z + 4)
                    # This maps (-4,-4)->0 to (4,4)->80
                    x_delta = np.clip(x_delta + 4, 0, 8)
                    z_delta = np.clip(z_delta + 4, 0, 8)
                    actions = (x_delta * 9 + z_delta).astype(np.int64)
                else:
                    # Ability usage: take column with most activity
                    actions = data[:, -13].astype(np.int64)  # Adjust based on data

                all_obs.append(obs)
                all_actions.append(actions)

            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

        # Concatenate
        self.observations = np.vstack(all_obs)
        self.actions = np.concatenate(all_actions)

        # Normalize observations
        self.obs_mean = self.observations.mean(axis=0)
        self.obs_std = self.observations.std(axis=0) + 1e-8
        self.observations = (self.observations - self.obs_mean) / self.obs_std

        print(f"\nDataset loaded:")
        print(f"  Total frames: {len(self.observations):,}")
        print(f"  Observation shape: {self.observations.shape}")
        print(f"  Action classes: {len(np.unique(self.actions))}")
        print(f"  Action distribution: {np.bincount(self.actions.astype(int))[:10]}...")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.observations[idx]),
            torch.LongTensor([self.actions[idx]])[0]
        )


class EzrealModel(nn.Module):
    """Neural network for Ezreal behavioral cloning."""

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


def train_model(
    data_dir="data/ezreal_data/NP",
    max_games=500,
    obs_cols=100,
    batch_size=512,
    epochs=30,
    learning_rate=0.001,
    device=None,
):
    """Train the Ezreal model."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = EzrealDataset(data_dir, max_games=max_games, obs_cols=obs_cols)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Create model
    input_dim = obs_cols
    output_dim = 81  # 9x9 movement grid

    model = EzrealModel(input_dim, output_dim).to(device)
    print(f"\nModel architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Output: {output_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for obs, actions in pbar:
            obs = obs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            outputs = model(obs)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        top5_correct = 0

        with torch.no_grad():
            for obs, actions in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)

                outputs = model(obs)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == actions).sum().item()
                total += len(actions)

                # Top-5 accuracy
                _, top5_preds = outputs.topk(5, dim=1)
                top5_correct += (top5_preds == actions.unsqueeze(1)).any(dim=1).sum().item()

        val_acc = correct / total
        top5_acc = top5_correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
            }, "ezreal_model_best.pt")

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Top5={top5_acc:.4f}, Best={best_val_acc:.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'obs_mean': dataset.obs_mean,
        'obs_std': dataset.obs_std,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
    }, "ezreal_model_final.pt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved: ezreal_model_best.pt, ezreal_model_final.pt")

    return model


def decode_action(action_id):
    """Decode action ID to movement direction."""
    x_delta = (action_id // 9) - 4  # -4 to 4
    z_delta = (action_id % 9) - 4   # -4 to 4
    return x_delta, z_delta


def visualize_predictions(model_path="ezreal_model_best.pt", data_dir="data/ezreal_data/NP"):
    """Visualize some predictions from the trained model."""
    import matplotlib.pyplot as plt

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = EzrealModel(checkpoint['input_dim'], checkpoint['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    obs_mean = checkpoint['obs_mean']
    obs_std = checkpoint['obs_std']

    # Load a sample game
    data_path = Path(data_dir)
    npy_files = list(data_path.glob("*.npy"))
    data = np.load(npy_files[0], allow_pickle=True).astype(np.float32)

    # Get observations
    obs = data[:100, :checkpoint['input_dim']]
    obs = (obs - obs_mean) / obs_std

    # Get actual actions
    x_delta = np.clip(data[:100, -17].astype(int) + 4, 0, 8)
    z_delta = np.clip(data[:100, -16].astype(int) + 4, 0, 8)
    actual = x_delta * 9 + z_delta

    # Predict
    with torch.no_grad():
        outputs = model(torch.FloatTensor(obs))
        predicted = torch.argmax(outputs, dim=1).numpy()

    accuracy = (predicted == actual).mean()
    print(f"Sample accuracy: {accuracy:.4f}")

    # Decode and show
    print("\nSample predictions (first 20):")
    print("Frame | Actual (x,z) | Predicted (x,z) | Match")
    print("-" * 50)
    for i in range(20):
        ax, az = decode_action(actual[i])
        px, pz = decode_action(predicted[i])
        match = "✓" if actual[i] == predicted[i] else "✗"
        print(f"  {i:3d} | ({ax:2d}, {az:2d})    | ({px:2d}, {pz:2d})       | {match}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Ezreal behavioral cloning model")
    parser.add_argument("--data_dir", default="data/ezreal_data/NP", help="Path to NP folder")
    parser.add_argument("--max_games", type=int, default=500, help="Max games to load")
    parser.add_argument("--obs_cols", type=int, default=100, help="Observation columns to use")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions after training")

    args = parser.parse_args()

    model = train_model(
        data_dir=args.data_dir,
        max_games=args.max_games,
        obs_cols=args.obs_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if args.visualize:
        visualize_predictions()
