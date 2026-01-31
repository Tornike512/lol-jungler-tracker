"""
Find Minimap Position
=====================
Interactive tool to find the correct minimap coordinates on your screen.
Run this and follow the instructions.
"""
import mss
import cv2
import numpy as np
import time

print("=" * 60)
print("MINIMAP FINDER TOOL")
print("=" * 60)
print("\nThis tool will help find the exact minimap position on your screen.")
print("\nInstructions:")
print("1. Make sure League is running and visible (in borderless/windowed)")
print("2. You should be in-game with the minimap visible")
print("3. Press Ctrl+C to stop the tool")
print("\nThe tool will capture the bottom-right corner of your screen")
print("where the minimap should be.\n")

input("Press Enter when ready (make sure League is visible)...")

print("\nWaiting 3 seconds for you to focus League...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)
print("  CAPTURING NOW!\n")

sct = mss.mss()

# Test different minimap positions for 1920x1080
# The minimap might be shifted up/down/left/right
test_configs = [
    {"name": "Default (100% HUD)", "left": 1647, "top": 817,
     "width": 267, "height": 267},
    {"name": "Shifted Up", "left": 1647, "top": 750, "width": 267, "height": 334},
    {"name": "Shifted Up More", "left": 1647,
        "top": 700, "width": 267, "height": 380},
    {"name": "Larger Area", "left": 1600, "top": 700, "width": 320, "height": 380},
    {"name": "Full Bottom Right", "left": 1550,
        "top": 650, "width": 370, "height": 430},
    {"name": "Very Large", "left": 1500, "top": 600, "width": 420, "height": 480},
]

print("\nTesting different minimap capture regions...")
print("Check the saved images to see which one contains the full minimap.\n")

for i, config in enumerate(test_configs):
    print(f"Capturing: {config['name']}")
    print(
        f"  Region: left={config['left']}, top={config['top']}, size={config['width']}x{config['height']}")

    try:
        screenshot = sct.grab(config)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        filename = f"minimap_test_{i}_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}.png"
        cv2.imwrite(filename, img)
        print(f"  Saved: {filename}")

        # Show image dimensions
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

    except Exception as e:
        print(f"  Error: {e}")

    print()

print("=" * 60)
print("Check the saved images above.")
print("The correct one should show the FULL minimap with:")
print("- Your champion icon (with green circle around it)")
print("- The full map including all 3 lanes")
print("- The jungle areas")
print("=" * 60)
print("\nTell me which image shows the full minimap correctly,")
print("and I'll update the coordinates in the script!")
