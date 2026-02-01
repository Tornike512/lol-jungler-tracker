"""
Complete Ezreal Training Script
===============================
Trains multiple neural networks to predict all aspects of Ezreal gameplay:
- Movement direction (x, z deltas)
- Ability usage (Q, W, E, R)
- Attack patterns
- Item usage

Uses all 2268 features from the TLoL dataset.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class EzrealCompleteDataset(Dataset):
    """Dataset that extracts ALL actions from Ezreal gameplay data."""

    def __init__(self, data_dir: str, max_games: int = None, use_all_features: bool = True):
        """
        Args:
            data_dir: Path to the NP folder with .npy files
            max_games: Max number of games to load
            use_all_features: If True, use all 2268 features, else use first 100
        """
        self.data_dir = Path(data_dir)
        self.npy_files = list(self.data_dir.glob("*.npy"))

        if max_games:
            self.npy_files = self.npy_files[:max_games]

        print(f"Loading {len(self.npy_files)} games...")

        # Load all games
        all_obs = []
        all_movement = []
        all_abilities = []

        for npy_file in tqdm(self.npy_files, desc="Loading games"):
            try:
                data = np.load(npy_file, allow_pickle=True).astype(np.float32)

                # Replace inf/nan with finite values
                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

                # Extract ALL observations (2268 features)
                if use_all_features:
                    obs = data[:, :]  # All columns
                else:
                    obs = data[:, :100]  # First 100 only

                # Extract movement (columns -17 and -16)
                x_delta = data[:, -17].astype(np.int32)
                z_delta = data[:, -16].astype(np.int32)

                # Clip and encode movement as class 0-80
                x_delta = np.clip(x_delta + 4, 0, 8)
                z_delta = np.clip(z_delta + 4, 0, 8)
                movement = (x_delta * 9 + z_delta).astype(np.int64)

                # Extract abilities from last 20 columns
                # Based on analysis: columns -20 to -1 contain action data
                abilities = data[:, -20:].astype(np.float32)

                all_obs.append(obs)
                all_movement.append(movement)
                all_abilities.append(abilities)

            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

        # Concatenate all data
        self.observations = np.vstack(all_obs)
        self.movement = np.concatenate(all_movement)
        self.abilities = np.vstack(all_abilities)

        # Normalize observations
        self.obs_mean = self.observations.mean(axis=0)
        self.obs_std = self.observations.std(axis=0) + 1e-8
        self.observations = (self.observations - self.obs_mean) / self.obs_std

        print(f"\nDataset loaded:")
        print(f"  Total frames: {len(self.observations):,}")
        print(f"  Observation shape: {self.observations.shape}")
        print(f"  Movement classes: {len(np.unique(self.movement))}")
        print(f"  Ability features: {self.abilities.shape[1]}")

        # Analyze ability columns
        print("\n  Ability column analysis:")
        for i in range(self.abilities.shape[1]):
            col = self.abilities[:, i]
            unique = np.unique(col)
            if len(unique) < 10:
                print(
                    f"    Col {i-20}: {len(unique)} unique values - {unique}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'obs': torch.FloatTensor(self.observations[idx]),
            'movement': torch.LongTensor([self.movement[idx]])[0],
            'abilities': torch.FloatTensor(self.abilities[idx]),
        }


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

        # Separate heads for each ability
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


def train_movement_model(dataset, epochs=30, batch_size=512, device='cpu'):
    """Train the movement prediction model."""
    print("\n" + "=" * 60)
    print("TRAINING MOVEMENT MODEL")
    print("=" * 60)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = dataset.observations.shape[1]
    model = MovementModel(input_dim).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3)

    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            obs = batch['obs'].to(device)
            movement = batch['movement'].to(device)

            optimizer.zero_grad()
            outputs = model(obs)
            loss = criterion(outputs, movement)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                movement = batch['movement'].to(device)

                outputs = model(obs)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == movement).sum().item()
                total += len(movement)

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'output_dim': 81,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
                'val_accuracy': val_acc,
            }, 'ezreal_movement_model.pt')
            print(f"  [Saved] New best accuracy: {val_acc:.2%}")

    return model, best_val_acc


def train_ability_model(dataset, epochs=30, batch_size=512, device='cpu'):
    """Train the ability usage prediction model."""
    print("\n" + "=" * 60)
    print("TRAINING ABILITY MODEL")
    print("=" * 60)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = dataset.observations.shape[1]
    num_abilities = dataset.abilities.shape[1]
    model = AbilityModel(input_dim, num_abilities).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Predicting {num_abilities} ability outputs")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            obs = batch['obs'].to(device)
            abilities = batch['abilities'].to(device)

            optimizer.zero_grad()
            outputs = model(obs)
            loss = criterion(outputs, abilities)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                abilities = batch['abilities'].to(device)

                outputs = model(obs)
                loss = criterion(outputs, abilities)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches

        print(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'num_abilities': num_abilities,
                'obs_mean': dataset.obs_mean,
                'obs_std': dataset.obs_std,
                'val_loss': avg_val_loss,
            }, 'ezreal_ability_model.pt')
            print(f"  [Saved] New best validation loss: {avg_val_loss:.4f}")

    return model, best_val_loss


def main():
    """Main training function."""
    print("=" * 60)
    print("EZREAL COMPLETE TRAINING")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset with ALL features
    data_dir = "data/ezreal_data/NP"

    print("\nLoading dataset with ALL features (2268 columns)...")
    dataset = EzrealCompleteDataset(
        data_dir, max_games=500, use_all_features=True)

    # Train movement model
    movement_model, movement_acc = train_movement_model(
        dataset, epochs=30, batch_size=512, device=device
    )

    print(f"\nMovement model best accuracy: {movement_acc:.2%}")

    # Train ability model
    ability_model, ability_loss = train_ability_model(
        dataset, epochs=30, batch_size=512, device=device
    )

    print(f"\nAbility model best validation loss: {ability_loss:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nSaved models:")
    print("  - ezreal_movement_model.pt")
    print("  - ezreal_ability_model.pt")


if __name__ == "__main__":
    main()
