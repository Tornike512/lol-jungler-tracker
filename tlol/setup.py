"""
TLoL Ezreal Dataset Setup
=========================
Downloads and prepares the Ezreal dataset for training.

Dataset: ~10,000 Ezreal early games, 5.4 million frames
Source: https://github.com/MiscellaneousStuff/tlol
"""

import os
import subprocess
import sys

# Dataset URLs
EZREAL_CONVERTED_URL = "https://drive.google.com/file/d/1xcGVd8kD98J9QxM866MSx-cELzxgT3GU/view?usp=sharing"
EZREAL_RAW_URL = "https://drive.google.com/file/d/1EXjJD1h9GNN4A8e7SaZB0HhRXcdmHeho/view?usp=sharing"

# Extract file IDs from Google Drive URLs
EZREAL_CONVERTED_ID = "1xcGVd8kD98J9QxM866MSx-cELzxgT3GU"
EZREAL_RAW_ID = "1EXjJD1h9GNN4A8e7SaZB0HhRXcdmHeho"


def install_dependencies():
    """Install required packages."""
    packages = [
        "gdown",           # Google Drive downloader
        "torch",           # PyTorch
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    ]

    print("Installing dependencies...")
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("Dependencies installed!\n")


def download_dataset(use_converted=True):
    """Download the Ezreal dataset from Google Drive."""
    import gdown

    os.makedirs("data", exist_ok=True)

    if use_converted:
        print("Downloading Ezreal dataset (Converted - Observations/Actions format)...")
        print("This is the recommended format for training.\n")
        file_id = EZREAL_CONVERTED_ID
        output = "data/ezreal_converted.zip"
    else:
        print("Downloading Ezreal dataset (Raw DBs)...")
        file_id = EZREAL_RAW_ID
        output = "data/ezreal_raw.zip"

    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, output, quiet=False)
        print(f"\nDownloaded to: {output}")
        print("\nNext step: Extract the archive and run explore_data.py")
        return output
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download instructions:")
        print(f"1. Go to: {EZREAL_CONVERTED_URL if use_converted else EZREAL_RAW_URL}")
        print("2. Click 'Download'")
        print(f"3. Save to: {os.path.abspath(output)}")
        return None


def main():
    print("=" * 60)
    print("TLoL Ezreal Dataset Setup")
    print("=" * 60)
    print()
    print("This will download the Ezreal dataset (~10,000 games)")
    print("containing 5.4 million frames for training.\n")

    # Install dependencies
    install_dependencies()

    # Download dataset
    download_dataset(use_converted=True)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Extract the downloaded zip file to data/ezreal/")
    print("2. Run: python explore_data.py")
    print("3. Run: python train_ezreal.py")


if __name__ == "__main__":
    main()
