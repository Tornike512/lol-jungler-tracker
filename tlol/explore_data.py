"""
Explore TLoL Ezreal Dataset
===========================
Analyzes the dataset structure and shows what features/actions are available.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_single_game(filepath):
    """Load a single game's pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def explore_dataset(data_dir="data/ezreal"):
    """Explore the dataset structure."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        print("\nPlease:")
        print("1. Run setup.py to download the dataset")
        print("2. Extract the zip to data/ezreal/")
        return

    # Find all pickle files
    pkl_files = list(data_path.glob("**/*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {data_path}")
        print("Make sure you extracted the dataset correctly.")
        return

    print("=" * 70)
    print("EZREAL DATASET EXPLORATION")
    print("=" * 70)
    print(f"\nFound {len(pkl_files)} game files\n")

    # Load first game to analyze structure
    print("Loading sample game...")
    sample_game = load_single_game(pkl_files[0])

    print(f"\nGame file: {pkl_files[0].name}")
    print(f"Type: {type(sample_game)}")

    if isinstance(sample_game, pd.DataFrame):
        analyze_dataframe(sample_game)
    elif isinstance(sample_game, dict):
        analyze_dict(sample_game)
    else:
        print(f"Data shape: {sample_game.shape if hasattr(sample_game, 'shape') else 'N/A'}")

    # Calculate total frames across sample
    print("\n" + "-" * 70)
    print("DATASET STATISTICS (sampling first 100 games)")
    print("-" * 70)

    sample_size = min(100, len(pkl_files))
    total_frames = 0
    frame_counts = []

    for pkl_file in pkl_files[:sample_size]:
        game = load_single_game(pkl_file)
        if isinstance(game, pd.DataFrame):
            frames = len(game)
        elif hasattr(game, '__len__'):
            frames = len(game)
        else:
            frames = 0
        total_frames += frames
        frame_counts.append(frames)

    avg_frames = total_frames / sample_size
    estimated_total = avg_frames * len(pkl_files)

    print(f"Games sampled: {sample_size}")
    print(f"Avg frames per game: {avg_frames:.0f}")
    print(f"Min frames: {min(frame_counts)}")
    print(f"Max frames: {max(frame_counts)}")
    print(f"Estimated total frames: {estimated_total:,.0f}")
    print(f"At 4 fps, ~{estimated_total / 4 / 60:.0f} minutes of gameplay")


def analyze_dataframe(df):
    """Analyze a pandas DataFrame game."""
    print(f"\nDataFrame shape: {df.shape}")
    print(f"  - Rows (frames): {df.shape[0]}")
    print(f"  - Columns (features): {df.shape[1]}")

    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Identify column categories
    columns = df.columns.tolist()

    # Separate observation and action columns
    obs_cols = [c for c in columns if not c.startswith("using_")]
    act_cols = [c for c in columns if c.startswith("using_")]

    print("\n" + "-" * 70)
    print("OBSERVATIONS (Input Features)")
    print("-" * 70)

    # Group by type
    time_cols = [c for c in obs_cols if "time" in c.lower() or "spawn" in c.lower()]
    player_cols = [c for c in obs_cols if "_0" in c]  # _0 typically means the player
    other_cols = [c for c in obs_cols if c not in time_cols and c not in player_cols]

    print(f"\nTime features ({len(time_cols)}):")
    for c in time_cols[:10]:
        print(f"  - {c}")

    print(f"\nPlayer state features ({len(player_cols)}):")

    # Categorize player features
    health_cols = [c for c in player_cols if "health" in c]
    position_cols = [c for c in player_cols if "position" in c]
    ability_cols = [c for c in player_cols if any(x in c for x in ["_level_", "_cd_", "_cast_"])]
    stat_cols = [c for c in player_cols if c not in health_cols + position_cols + ability_cols]

    print(f"\n  Health/Mana ({len(health_cols)}):")
    for c in health_cols[:5]:
        print(f"    - {c}: {df[c].iloc[0]:.2f}")

    print(f"\n  Position ({len(position_cols)}):")
    for c in position_cols[:5]:
        print(f"    - {c}: {df[c].iloc[0]:.2f}")

    print(f"\n  Abilities ({len(ability_cols)}):")
    for c in ability_cols[:10]:
        print(f"    - {c}")

    print(f"\n  Other stats ({len(stat_cols)}):")
    for c in stat_cols[:10]:
        print(f"    - {c}")

    print("\n" + "-" * 70)
    print("ACTIONS (Output Labels)")
    print("-" * 70)

    print(f"\nAction columns ({len(act_cols)}):")
    for c in act_cols:
        if c in df.columns:
            action_count = (df[c] == 1.0).sum()
            pct = action_count / len(df) * 100
            print(f"  - {c}: {action_count} uses ({pct:.1f}%)")

    # Show movement action encoding if present
    move_cols = [c for c in columns if "delta" in c.lower() or "move" in c.lower()]
    if move_cols:
        print(f"\nMovement features ({len(move_cols)}):")
        for c in move_cols[:5]:
            print(f"  - {c}")

    print("\n" + "-" * 70)
    print("SAMPLE DATA (First 5 frames)")
    print("-" * 70)
    print(df.head())


def analyze_dict(data):
    """Analyze a dictionary game format."""
    print(f"\nDictionary keys: {list(data.keys())}")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  - {key}: len={len(value)}")
        else:
            print(f"  - {key}: {type(value)}")


if __name__ == "__main__":
    # Allow custom data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ezreal"
    explore_dataset(data_dir)
