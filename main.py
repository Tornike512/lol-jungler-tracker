"""
League of Legends Jungler Tracker - Simple Edition

Shows if enemy jungler is visible on minimap.
"""
import sys
import time
import ctypes
import mss
import numpy as np
import cv2
from typing import Optional

from config import config, MinimapCalibration
from detector import detect_enemy_jungler, is_game_active, grid_detector
from overlay import tracker_overlay


class SimpleJunglerTracker:
    """Simple tracker - just shows if jungler is visible."""

    def __init__(self, test_mode: bool = False):
        self._running = False
        self._enemy_jungler = ""
        self._sct = mss.mss()
        self._minimap_config: Optional[MinimapCalibration] = None
        self._test_mode = test_mode

    def start(self):
        """Start the tracker."""
        print("=" * 50)
        if self._test_mode:
            print("  JUNGLER TRACKER - TEST MODE")
        else:
            print("  JUNGLER TRACKER - Simple Mode")
        print("=" * 50)
        print()

        # Get screen resolution
        monitor = self._sct.monitors[1]
        screen_w, screen_h = monitor['width'], monitor['height']
        print(f"[INFO] Screen resolution: {screen_w}x{screen_h}")

        # Ask user for minimap side
        print("\nWhere is your minimap?")
        print("1. Left side")
        print("2. Right side (default)")
        try:
            choice = input("[INPUT] Enter 1 or 2 (or left/right): ").strip().lower()
            if choice in ["1", "left", "l"]:
                side = "left"
            else:
                side = "right"  # Default to right (most common)
        except:
            side = "right"

        # Ask for minimap scale
        print("\nWhat is your minimap scale in League settings? (default 100)")
        try:
            scale_input = input("[INPUT] Enter scale (0-100): ").strip()
            if scale_input:
                minimap_scale = int(scale_input)
            else:
                minimap_scale = 100
        except:
            minimap_scale = 100

        # Calculate minimap position with scale
        self._minimap_config = MinimapCalibration.from_resolution(screen_w, screen_h, side, minimap_scale)
        print(f"[INFO] Minimap: x={self._minimap_config.x}, y={self._minimap_config.y}, "
              f"size={self._minimap_config.width}x{self._minimap_config.height}")

        self._running = True

        # Start overlay
        tracker_overlay.start()
        tracker_overlay.update_status("Waiting for game...")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[STATUS] Stopped by user")
        finally:
            self.stop()

    def _main_loop(self):
        """Main loop."""
        while self._running:
            if self._test_mode:
                # TEST MODE: Skip API, ask for champion directly
                self._enemy_jungler = self._manual_jungler_input()
                if not self._enemy_jungler:
                    print("[ERROR] No champion entered")
                    continue

                print(f"[TEST] Will track: {self._enemy_jungler}")
                tracker_overlay.update_jungler(self._enemy_jungler)

                # Set up the grid detector
                if not grid_detector.set_champion(self._enemy_jungler):
                    print(f"[ERROR] Failed to load icon for {self._enemy_jungler}")
                    tracker_overlay.update_status("Failed to load champion icon")
                    time.sleep(2)
                    continue

                # Run tracking loop (in test mode, don't check game API)
                self._tracking_loop_test()
                continue

            # NORMAL MODE: Wait for game
            if not is_game_active():
                tracker_overlay.update_status("Waiting for game...")
                tracker_overlay.update_jungler("")
                tracker_overlay.set_visible(False)
                time.sleep(2)
                continue

            # Game detected
            print("[STATUS] Game detected!")
            tracker_overlay.update_status("Detecting jungler...")

            # Get enemy jungler from API
            self._enemy_jungler = self._wait_for_jungler()
            if not self._enemy_jungler:
                print("[WARNING] Could not detect jungler")
                tracker_overlay.update_status("Could not detect jungler")
                time.sleep(5)
                continue

            print(f"[STATUS] Tracking: {self._enemy_jungler}")
            tracker_overlay.update_jungler(self._enemy_jungler)

            # Set up the grid detector with the champion icon
            if not grid_detector.set_champion(self._enemy_jungler):
                print(f"[ERROR] Failed to load icon for {self._enemy_jungler}")
                tracker_overlay.update_status("Failed to load champion icon")
                time.sleep(5)
                continue

            # Main tracking loop
            self._tracking_loop()

    def _wait_for_jungler(self) -> str:
        """Wait for API to return jungler info and confirm with user."""
        for _ in range(15):  # Try for 30 seconds
            if not is_game_active():
                return ""
            jungler = detect_enemy_jungler()
            if jungler:
                # Ask user to confirm the detected jungler
                return self._confirm_jungler(jungler)
            time.sleep(2)

        # If API couldn't detect, ask user to input manually
        return self._manual_jungler_input()

    def _confirm_jungler(self, detected_jungler: str) -> str:
        """Ask user to confirm the detected jungler."""
        print(f"\n[DETECTED] Enemy jungler appears to be: {detected_jungler}")
        print("Is this correct?")
        print("1. Yes (press Enter)")
        print("2. No, let me type the correct champion")

        try:
            choice = input("[INPUT] Enter choice (1 or 2): ").strip()
            if choice == "2":
                return self._manual_jungler_input()
            return detected_jungler
        except:
            return detected_jungler

    def _manual_jungler_input(self) -> str:
        """Ask user to manually input the enemy jungler name."""
        print("\n[INPUT] Enter the enemy jungler's champion name:")
        try:
            jungler = input("> ").strip()
            if jungler:
                return jungler
        except:
            pass
        return ""

    def _capture_minimap(self) -> Optional[np.ndarray]:
        """Capture minimap region."""
        if not self._minimap_config:
            return None
        try:
            region = {
                "left": self._minimap_config.x,
                "top": self._minimap_config.y,
                "width": self._minimap_config.width,
                "height": self._minimap_config.height
            }
            screenshot = self._sct.grab(region)
            return np.array(screenshot)[:, :, :3]  # BGR
        except Exception as e:
            print(f"[ERROR] Capture failed: {e}")
            return None

    def _detect_enemy_visible(self, minimap: np.ndarray) -> bool:
        """
        Detect if enemy champion icon is visible on minimap.

        Enemy icons have a red circular border. We look for:
        - Red color in HSV space
        - Circular shape
        - Appropriate size for champion icons
        - Not at the very edges (those are usually UI elements)
        """
        if minimap is None:
            return False

        height, width = minimap.shape[:2]

        # Crop out edges (UI elements, not actual minimap)
        margin = int(width * 0.08)
        cropped = minimap[margin:height-margin, margin:width-margin]

        if cropped.size == 0:
            return False

        # Convert to HSV
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Red color detection (red wraps around in HSV)
        lower_red1 = np.array([0, 150, 100])  # More strict saturation
        upper_red1 = np.array([8, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([172, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for champion icon sized circles
        min_icon_area = 80   # Minimum area for champion icon
        max_icon_area = 800  # Maximum area

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_icon_area < area < max_icon_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # Champion icons are fairly circular (circularity > 0.5)
                    if circularity > 0.5:
                        return True

        return False

    def _tracking_loop(self):
        """Main tracking loop using grid-based template matching."""
        last_visible = False
        visible_frames = 0
        invisible_frames = 0

        # Require multiple frames to confirm state change (reduces flickering)
        CONFIRM_FRAMES = 3

        print(f"[STATUS] Now tracking {self._enemy_jungler} on minimap...")
        tracker_overlay.update_status("Tracking...")

        while self._running and is_game_active():
            minimap = self._capture_minimap()
            if minimap is None:
                time.sleep(0.1)
                continue

            # Use grid-based template matching
            is_visible, cell_position = grid_detector.detect(minimap)

            # Debounce detection
            if is_visible:
                visible_frames += 1
                invisible_frames = 0
            else:
                invisible_frames += 1
                visible_frames = 0

            # Only change state after consistent detection
            if visible_frames >= CONFIRM_FRAMES and not last_visible:
                # Just appeared
                last_visible = True
                tracker_overlay.set_visible(True)
                tracker_overlay.update_status("ENEMY JUNGLER APPEARED")
                tracker_overlay.show_alert(f"⚠ {self._enemy_jungler} APPEARED!")
                print(f"[APPEARED] {self._enemy_jungler} detected on minimap!")

            elif invisible_frames >= CONFIRM_FRAMES and last_visible:
                # Just disappeared
                last_visible = False
                tracker_overlay.set_visible(False)
                tracker_overlay.update_status("Enemy jungler not detected")
                print(f"[GONE] {self._enemy_jungler} no longer visible")

            time.sleep(0.066)  # ~15 fps

        print("[STATUS] Game ended")
        tracker_overlay.update_status("Game ended")
        grid_detector.reset()

    def _tracking_loop_test(self):
        """Test tracking loop - doesn't require game API."""
        last_visible = False
        visible_frames = 0
        invisible_frames = 0
        CONFIRM_FRAMES = 2  # Reduced for faster response

        print(f"\n[TEST] Now tracking {self._enemy_jungler} on minimap...")
        print("[TEST] Press Ctrl+C to stop and test another champion")
        print("[TEST] Confidence scores: appear > 0.15, disappear < 0.10")
        print("[TEST] Debug images saved every 5 seconds to: lol-jungler-tracker/debug/")
        print("-" * 50)
        tracker_overlay.update_status("Tracking (TEST MODE)")

        last_print_time = 0
        last_debug_save = 0
        frame_count = 0

        try:
            while self._running:
                minimap = self._capture_minimap()
                if minimap is None:
                    time.sleep(0.1)
                    continue

                # Use grid-based template matching
                is_visible, cell_position = grid_detector.detect(minimap)
                confidence = grid_detector.get_last_confidence()

                frame_count += 1

                # Print confidence every ~1 second for debugging
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    status = "DETECTED!" if is_visible else "not found"
                    print(f"[TEST] Confidence: {confidence:.3f} | {self._enemy_jungler}: {status}")
                    last_print_time = current_time

                # Save debug images every 10 seconds
                if current_time - last_debug_save >= 10.0:
                    grid_detector.save_debug_image(minimap, "test")
                    last_debug_save = current_time

                # Debounce detection
                if is_visible:
                    visible_frames += 1
                    invisible_frames = 0
                else:
                    invisible_frames += 1
                    visible_frames = 0

                # Only change state after consistent detection
                if visible_frames >= CONFIRM_FRAMES and not last_visible:
                    last_visible = True
                    tracker_overlay.set_visible(True)
                    tracker_overlay.update_status("ENEMY JUNGLER APPEARED")
                    tracker_overlay.show_alert(f"⚠ {self._enemy_jungler} APPEARED!")
                    print(f"[TEST DETECTED] {self._enemy_jungler} found! (confidence: {confidence:.3f})")

                elif invisible_frames >= CONFIRM_FRAMES and last_visible:
                    last_visible = False
                    tracker_overlay.set_visible(False)
                    tracker_overlay.update_status("Enemy jungler not detected")
                    print(f"[TEST GONE] {self._enemy_jungler} no longer visible (confidence: {confidence:.3f})")

                time.sleep(0.066)  # ~15 fps

        except KeyboardInterrupt:
            print("\n[TEST] Stopping test...")
            grid_detector.reset()

    def stop(self):
        """Stop tracker."""
        self._running = False
        tracker_overlay.stop()
        print("[STATUS] Goodbye!")


def main():
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        pass

    # Check for test mode
    test_mode = "--test" in sys.argv or "-t" in sys.argv

    tracker = SimpleJunglerTracker(test_mode=test_mode)
    tracker.start()


if __name__ == "__main__":
    main()
