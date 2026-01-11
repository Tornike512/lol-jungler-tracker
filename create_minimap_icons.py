"""
Create minimap-style circular icons from Data Dragon square icons.
Minimap icons are circular with a colored border.
"""
import cv2
import numpy as np
import os
from config import CHAMPIONS_DIR


def create_circular_icon(input_path: str, output_path: str, size: int = 24, border_color=(0, 0, 255)):
    """
    Create a circular minimap-style icon from a square icon.

    Args:
        input_path: Path to square icon
        output_path: Path to save circular icon
        size: Output size in pixels
        border_color: BGR color for border (default red for enemy)
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        return False

    # Resize to target size
    img = cv2.resize(img, (size, size))

    # Create circular mask
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = center - 2  # Leave room for border
    cv2.circle(mask, (center, center), radius, 255, -1)

    # Apply mask
    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[mask == 255] = img[mask == 255]

    # Add colored border
    cv2.circle(result, (center, center), radius + 1, border_color, 2)

    cv2.imwrite(output_path, result)
    return True


def process_champion(champion_name: str):
    """Create minimap icons for a champion."""
    champion_dir = os.path.join(CHAMPIONS_DIR, champion_name)
    source_icon = os.path.join(champion_dir, f"{champion_name}.png")

    if not os.path.exists(source_icon):
        print(f"[ERROR] Source icon not found: {source_icon}")
        return False

    # Create circular versions at different sizes
    sizes = [20, 22, 24, 26, 28, 30, 32]

    for size in sizes:
        output_path = os.path.join(champion_dir, f"{champion_name}_circle_{size}.png")
        create_circular_icon(source_icon, output_path, size)
        print(f"  Created {size}px circular icon")

    # Also create versions with different border thicknesses
    for size in [24, 28]:
        output_path = os.path.join(champion_dir, f"{champion_name}_circle_{size}_thick.png")
        img = cv2.imread(source_icon)
        img = cv2.resize(img, (size, size))

        mask = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        radius = center - 3
        cv2.circle(mask, (center, center), radius, 255, -1)

        result = np.zeros((size, size, 3), dtype=np.uint8)
        result[mask == 255] = img[mask == 255]
        cv2.circle(result, (center, center), radius + 2, (0, 0, 255), 3)

        cv2.imwrite(output_path, result)
        print(f"  Created {size}px thick border icon")

    print(f"[OK] Created minimap icons for {champion_name}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        champion = sys.argv[1]
    else:
        # Process all champions in assets folder
        for champion in os.listdir(CHAMPIONS_DIR):
            champion_path = os.path.join(CHAMPIONS_DIR, champion)
            if os.path.isdir(champion_path):
                print(f"\nProcessing {champion}...")
                process_champion(champion)
