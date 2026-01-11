"""
Screen capture and minimap detection for League of Legends.
Handles resolution-independent minimap location detection.

DEBUG MODE: Saves calibration images and minimap captures.
"""
import numpy as np
import cv2
import mss
import time
import os
from datetime import datetime
from typing import Optional, Tuple
from config import config, MinimapCalibration, BASE_DIR


# Debug directory
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


def debug_log(message: str):
    """Print debug message with timestamp."""
    if config.settings.debug_mode:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG {timestamp}] {message}")


class ScreenCapture:
    """Handles screen capture and minimap detection."""

    def __init__(self):
        self.sct = mss.mss()
        self._minimap_template = None
        self._last_calibration_time = 0
        self._calibration_interval = 300  # Only re-calibrate every 5 minutes (minimap doesn't move)
        self._capture_count = 0
        self._validation_fail_count = 0
        self._max_validation_fails = 10  # Only recalibrate after 10 consecutive failures

    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture the entire primary screen."""
        try:
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)
            img = np.array(screenshot)[:, :, :3]  # Remove alpha channel, BGR format
            debug_log(f"Screen captured: {img.shape[1]}x{img.shape[0]}")
            return img
        except Exception as e:
            print(f"[ERROR] Screen capture failed: {e}")
            debug_log(f"Screen capture error: {e}")
            return None

    def capture_minimap(self) -> Optional[np.ndarray]:
        """Capture just the minimap region."""
        if not config.minimap.is_valid:
            debug_log("Minimap capture failed: calibration invalid")
            return None

        try:
            region = {
                "left": config.minimap.x,
                "top": config.minimap.y,
                "width": config.minimap.width,
                "height": config.minimap.height
            }
            screenshot = self.sct.grab(region)
            minimap = np.array(screenshot)[:, :, :3]

            self._capture_count += 1

            # Debug: save minimap periodically
            if config.settings.debug_mode and self._capture_count % 75 == 1:  # Every 5 seconds at 15fps
                timestamp = datetime.now().strftime("%H%M%S")
                path = os.path.join(DEBUG_DIR, f"minimap_capture_{timestamp}.png")
                cv2.imwrite(path, minimap)
                debug_log(f"Saved minimap capture: {path}")

            return minimap
        except Exception as e:
            print(f"[ERROR] Minimap capture failed: {e}")
            debug_log(f"Minimap capture error: {e}")
            return None

    def detect_minimap(self, screen: np.ndarray) -> Optional[MinimapCalibration]:
        """
        Detect the minimap location on screen.
        Uses multiple detection strategies:
        1. Look for the characteristic minimap border/frame
        2. Detect the circular minimap background
        3. Fall back to corner detection (standard positions)
        """
        if screen is None:
            return None

        height, width = screen.shape[:2]
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        debug_log(f"Detecting minimap in screen {width}x{height}")

        # Strategy 1: Check standard minimap positions (corners)
        # Minimap is typically 200-280 pixels depending on settings
        minimap_sizes = [280, 260, 240, 220, 200, 180, 160, 300, 320]

        # Check bottom-right corner first (most common)
        for size in minimap_sizes:
            # Bottom-right
            x_pos = width - size - 10
            y_pos = height - size - 10
            roi = self._check_minimap_region(screen, gray, x_pos, y_pos, size)
            if roi is not None:
                debug_log(f"Found minimap at bottom-right: ({x_pos}, {y_pos}) size={size}")
                return MinimapCalibration(
                    x=x_pos,
                    y=y_pos,
                    width=size,
                    height=size,
                    side="right"
                )

            # Bottom-left (for players who moved minimap)
            x_pos = 10
            y_pos = height - size - 10
            roi = self._check_minimap_region(screen, gray, x_pos, y_pos, size)
            if roi is not None:
                debug_log(f"Found minimap at bottom-left: ({x_pos}, {y_pos}) size={size}")
                return MinimapCalibration(
                    x=x_pos,
                    y=y_pos,
                    width=size,
                    height=size,
                    side="left"
                )

        debug_log("Standard position detection failed, trying edge detection...")

        # Strategy 2: Use edge detection to find minimap border
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            x, y, w, h = cv2.boundingRect(contour)
            # Minimap should be roughly square and a reasonable size
            if 150 < w < 350 and 150 < h < 350 and 0.85 < w/h < 1.15:
                # Check if it's in a corner (likely minimap position)
                if (x < 50 or x > width - 400) and y > height - 400:
                    debug_log(f"Found minimap via contour: ({x}, {y}) size={w}x{h}")
                    return MinimapCalibration(
                        x=x, y=y, width=w, height=h,
                        side="left" if x < width // 2 else "right"
                    )

        debug_log("Minimap detection failed!")
        return None

    def _check_minimap_region(
        self, screen: np.ndarray, gray: np.ndarray,
        x: int, y: int, size: int
    ) -> Optional[np.ndarray]:
        """
        Check if a region looks like a minimap.
        Minimaps have characteristic features:
        - Distinct color patterns (green for map, gray for fog)
        - Relatively high contrast
        - Presence of the map terrain texture
        """
        try:
            if x < 0 or y < 0:
                return None
            if x + size > screen.shape[1] or y + size > screen.shape[0]:
                return None

            roi = screen[y:y+size, x:x+size]
            roi_gray = gray[y:y+size, x:x+size]

            # Check for minimap characteristics
            # 1. Should have some green tones (grass on map)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Green mask
            lower_green = np.array([35, 30, 30])
            upper_green = np.array([85, 255, 200])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.sum(green_mask > 0) / (size * size)

            # 2. Should have gray/dark areas (fog of war)
            lower_gray = np.array([0, 0, 20])
            upper_gray = np.array([180, 50, 100])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
            gray_ratio = np.sum(gray_mask > 0) / (size * size)

            # 3. Check contrast (minimap has distinct features)
            contrast = np.std(roi_gray)

            if config.settings.debug_mode:
                debug_log(f"  Region ({x},{y}) size={size}: green={green_ratio:.2f}, gray={gray_ratio:.2f}, contrast={contrast:.1f}")

            # Minimap should have:
            # - Some green (5-40%)
            # - Some gray/dark areas (10-60%)
            # - Reasonable contrast
            if 0.03 < green_ratio < 0.5 and 0.05 < gray_ratio < 0.7 and contrast > 25:
                return roi

            return None
        except Exception as e:
            debug_log(f"  Region check error: {e}")
            return None

    def calibrate(self) -> bool:
        """
        Perform minimap calibration.
        Returns True if calibration successful.
        """
        print("[STATUS] Starting minimap calibration...")

        screen = self.capture_screen()
        if screen is None:
            print("[ERROR] Failed to capture screen for calibration")
            return False

        # Debug: Save the full screen for analysis
        if config.settings.debug_mode:
            timestamp = datetime.now().strftime("%H%M%S")
            screen_path = os.path.join(DEBUG_DIR, f"fullscreen_{timestamp}.png")
            cv2.imwrite(screen_path, screen)
            debug_log(f"Saved fullscreen: {screen_path}")

        calibration = self.detect_minimap(screen)
        if calibration is None:
            print("[ERROR] Could not detect minimap. Make sure League of Legends is running.")
            return False

        config.minimap = calibration
        config.save_calibration()

        # Debug: Save the detected minimap region
        if config.settings.debug_mode:
            minimap_region = screen[calibration.y:calibration.y+calibration.height,
                                    calibration.x:calibration.x+calibration.width]
            minimap_path = os.path.join(DEBUG_DIR, f"calibrated_minimap_{timestamp}.png")
            cv2.imwrite(minimap_path, minimap_region)
            debug_log(f"Saved calibrated minimap: {minimap_path}")

        print(f"[STATUS] Minimap found at ({calibration.x}, {calibration.y}), "
              f"size: {calibration.width}x{calibration.height}, side: {calibration.side}")
        self._last_calibration_time = time.time()
        return True

    def should_recalibrate(self) -> bool:
        """Check if we should re-run calibration."""
        if not config.minimap.is_valid:
            return True
        return time.time() - self._last_calibration_time > self._calibration_interval

    def validate_calibration(self) -> bool:
        """
        Validate that current calibration is still correct.
        Quick check without full re-calibration.

        Uses consecutive failure counting to avoid recalibrating when
        Tab screen or other UI overlays temporarily cover the minimap.
        """
        if not config.minimap.is_valid:
            return False

        minimap = self.capture_minimap()
        if minimap is None:
            return False

        # Quick validation: check if the region still looks like a minimap
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 200])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (minimap.shape[0] * minimap.shape[1])

        is_valid = 0.03 < green_ratio < 0.5

        if is_valid:
            # Reset failure count on success
            self._validation_fail_count = 0
            return True
        else:
            # Track consecutive failures
            self._validation_fail_count += 1
            if self._validation_fail_count <= self._max_validation_fails:
                # Don't trigger recalibration yet - could be tab screen
                debug_log(f"Validation check failed ({self._validation_fail_count}/{self._max_validation_fails}): green_ratio={green_ratio:.2f} - skipping recalibration")
                return True  # Pretend it's valid to avoid recalibration spam
            else:
                debug_log(f"Calibration validation failed after {self._validation_fail_count} attempts: green_ratio={green_ratio:.2f}")
                self._validation_fail_count = 0  # Reset for next time
                return False


# Global screen capture instance
screen_capture = ScreenCapture()
