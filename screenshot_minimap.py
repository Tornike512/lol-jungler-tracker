"""
Minimap Screenshot Script
=========================
Takes a screenshot of the minimap after 3 seconds.
Run this while League is running to capture the minimap region.
"""
import time
from minimap_tracker import MinimapTracker
import cv2


def main():
    print("=" * 60)
    print("MINIMAP SCREENSHOT TOOL")
    print("=" * 60)
    print("\nMake sure League is running and visible!")
    print("The script will capture the minimap in 3 seconds...")
    print()

    # 3 second countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  CAPTURING!")

    # Initialize tracker and capture
    tracker = MinimapTracker()

    # Capture the minimap
    img = tracker.capture_minimap()

    # Find player position for debugging
    minimap_pos = tracker.find_player_position(img)

    # Create debug image with overlay
    debug_img = img.copy()

    if minimap_pos:
        # Draw circle at detected position
        cv2.circle(debug_img, minimap_pos, 12, (0, 255, 0), 2)
        cv2.circle(debug_img, minimap_pos, 3, (0, 255, 0), -1)
        game_x, game_y = tracker.minimap_to_game_coords(
            minimap_pos[0], minimap_pos[1])
        cv2.putText(debug_img, f"({game_x}, {game_y})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        print(f"\n[+] Player detected at: ({game_x}, {game_y})")
        print(f"[+] Minimap pixel: {minimap_pos}")
    else:
        cv2.putText(debug_img, "Position not found", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        print("\n[!] Player position not detected")

    # Save both images
    cv2.imwrite("minimap_screenshot.png", img)
    cv2.imwrite("minimap_screenshot_debug.png", debug_img)

    print("\n[+] Saved:")
    print("    - minimap_screenshot.png (raw capture)")
    print("    - minimap_screenshot_debug.png (with detection overlay)")
    print("\nDone!")


if __name__ == "__main__":
    main()
