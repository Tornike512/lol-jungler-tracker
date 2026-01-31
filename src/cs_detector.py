"""
CS (Creep Score) Detection Module
Uses OCR to read the CS counter from the game screen and track changes.
"""
import time
import re
from typing import Optional, Tuple
import numpy as np
import cv2

# Lazy import easyocr to avoid slow startup
_reader = None


def get_ocr_reader():
    """Lazy initialization of OCR reader"""
    global _reader
    if _reader is None:
        try:
            import easyocr
            print("Initializing OCR reader (first time may take a moment)...")
            _reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print("OCR reader initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize OCR: {e}")
            print("CS detection will be disabled")
            return None
    return _reader


class CSDetector:
    """
    Detects and tracks CS (Creep Score) from game screen.

    The CS counter in LoL appears in several places:
    1. Tab scoreboard (top area when Tab is held)
    2. Near the minimap/HUD area

    For Practice Tool, we'll read from a fixed screen region.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        update_interval: float = 0.5  # Only run OCR every 0.5 seconds
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.update_interval = update_interval

        # CS tracking
        self.current_cs = 0
        self.previous_cs = 0
        self.cs_history = []
        self.max_history = 100

        # Timing
        self.last_update_time = 0.0
        self.last_cs_change_time = 0.0

        # Performance tracking
        self.ocr_time_ms = 0.0
        self.successful_reads = 0
        self.failed_reads = 0

        # Define CS counter region based on resolution
        # This is for the top-right stats area (visible when Tab is pressed or in HUD)
        # You may need to adjust these values based on your game settings
        self._setup_cs_region()

        print(f"CS Detector initialized")
        print(f"  Screen: {screen_width}x{screen_height}")
        print(f"  CS Region: {self.cs_region}")
        print(f"  Update interval: {update_interval}s")

    def _setup_cs_region(self):
        """
        Setup the screen region where CS counter appears.

        In LoL, the CS is shown:
        - Tab scoreboard: Top center-left area
        - HUD: Near the player stats (varies by skin)

        For 1920x1080, the Tab scoreboard CS is around:
        - Your row is typically in the upper portion
        - CS column is after K/D/A

        We'll use a region that captures the CS from the Tab scoreboard.
        The player needs to hold Tab briefly for accurate reading.

        Alternative: Read from the minion score icon near bottom-right HUD.
        """
        # Method 1: Tab scoreboard (requires Tab to be pressed)
        # This is more accurate but requires Tab input

        # Method 2: Bottom HUD minion counter (always visible)
        # Located near the minimap, shows minion icon with number
        # For 1920x1080: approximately x=1650-1750, y=750-800

        # We'll use the HUD method since it's always visible
        # Adjust these values if they don't match your setup

        if self.screen_width == 1920 and self.screen_height == 1080:
            # 1080p settings - CS counter in top-right scoreboard
            # User provided: x=1720 to x=1800, y=13
            self.cs_region = {
                "left": 1720,
                "top": 8,
                "width": 80,
                "height": 20
            }
        elif self.screen_width == 2560 and self.screen_height == 1440:
            # 1440p settings - scale from 1080p
            self.cs_region = {
                "left": 527,
                "top": 1000,
                "width": 60,
                "height": 33
            }
        else:
            # Scale from 1080p
            scale_x = self.screen_width / 1920
            scale_y = self.screen_height / 1080
            self.cs_region = {
                "left": int(395 * scale_x),
                "top": int(750 * scale_y),
                "width": int(45 * scale_x),
                "height": int(25 * scale_y)
            }

    def capture_cs_region(self) -> Optional[np.ndarray]:
        """Capture the CS counter region from screen"""
        try:
            import mss
            with mss.mss() as sct:
                screenshot = sct.grab(self.cs_region)
                img = np.array(screenshot)
                # Convert BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            print(f"Warning: Failed to capture CS region: {e}")
            return None

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        The CS counter in LoL is typically white/yellow text on dark background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # Threshold to get white text
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Slight dilation to make digits clearer
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Scale up for better OCR
        scale = 3
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return thresh

    def read_cs_from_image(self, img: np.ndarray) -> Optional[int]:
        """
        Read CS number from image using OCR.

        Returns:
            CS count if successfully read, None otherwise
        """
        reader = get_ocr_reader()
        if reader is None:
            return None

        try:
            start_time = time.time()

            # Preprocess
            processed = self.preprocess_image(img)

            # Run OCR
            results = reader.readtext(processed, detail=0, paragraph=False)

            self.ocr_time_ms = (time.time() - start_time) * 1000

            # Parse results - look for numbers
            for text in results:
                # Clean the text - keep only digits
                digits = re.sub(r'[^0-9]', '', text)
                if digits:
                    cs = int(digits)
                    # Sanity check - CS should be reasonable (0-999)
                    if 0 <= cs < 1000:
                        self.successful_reads += 1
                        return cs

            self.failed_reads += 1
            return None

        except Exception as e:
            self.failed_reads += 1
            return None

    def update(self) -> Tuple[int, int]:
        """
        Update CS reading from screen.

        Returns:
            (current_cs, cs_gained) - Current CS and CS gained since last update
        """
        current_time = time.time()

        # Rate limit OCR calls
        if current_time - self.last_update_time < self.update_interval:
            return self.current_cs, 0

        self.last_update_time = current_time

        # Capture and read CS
        img = self.capture_cs_region()
        if img is None:
            return self.current_cs, 0

        new_cs = self.read_cs_from_image(img)

        if new_cs is not None:
            self.previous_cs = self.current_cs

            # Only update if CS increased (can't decrease)
            if new_cs >= self.current_cs:
                self.current_cs = new_cs

                # Track history
                self.cs_history.append({
                    "time": current_time,
                    "cs": new_cs
                })
                if len(self.cs_history) > self.max_history:
                    self.cs_history.pop(0)

                if new_cs > self.previous_cs:
                    self.last_cs_change_time = current_time

        cs_gained = max(0, self.current_cs - self.previous_cs)
        return self.current_cs, cs_gained

    def get_cs_per_minute(self) -> float:
        """Calculate CS per minute based on history"""
        if len(self.cs_history) < 2:
            return 0.0

        first = self.cs_history[0]
        last = self.cs_history[-1]

        time_diff = last["time"] - first["time"]
        cs_diff = last["cs"] - first["cs"]

        if time_diff <= 0:
            return 0.0

        # Convert to per minute
        cs_per_minute = (cs_diff / time_diff) * 60.0
        return cs_per_minute

    def get_stats(self) -> dict:
        """Get CS detector statistics"""
        return {
            "current_cs": self.current_cs,
            "cs_per_minute": self.get_cs_per_minute(),
            "successful_reads": self.successful_reads,
            "failed_reads": self.failed_reads,
            "ocr_time_ms": self.ocr_time_ms,
            "accuracy": self.successful_reads / max(1, self.successful_reads + self.failed_reads)
        }

    def reset(self):
        """Reset CS tracking for new game/episode"""
        self.current_cs = 0
        self.previous_cs = 0
        self.cs_history = []
        self.last_cs_change_time = 0.0


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing CS Detector")
    print("=" * 60)
    print("Make sure League of Legends is running and visible")
    print("The CS counter should be visible on screen")
    print()

    detector = CSDetector()

    print("Starting CS detection test (30 seconds)...")
    print("Try getting some CS to see if detection works")
    print()

    start_time = time.time()
    last_print_time = 0

    try:
        while time.time() - start_time < 30:
            current_cs, cs_gained = detector.update()

            # Print every second
            if time.time() - last_print_time >= 1.0:
                stats = detector.get_stats()
                print(f"CS: {current_cs} | "
                      f"CS/min: {stats['cs_per_minute']:.1f} | "
                      f"OCR time: {stats['ocr_time_ms']:.0f}ms | "
                      f"Accuracy: {stats['accuracy']:.1%}")
                last_print_time = time.time()

            if cs_gained > 0:
                print(f"  >>> +{cs_gained} CS!")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user")

    print()
    print("Final Stats:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
