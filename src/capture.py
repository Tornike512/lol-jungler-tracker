"""
High-Performance Screen Capture Module
Implements GPU-accelerated frame grabbing with triple buffering and ROI extraction.
"""
import time
import threading
from queue import Queue
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import mss
import cv2

from .config import capture_cfg


@dataclass
class CaptureRegion:
    """Defines a screen region to capture"""
    name: str
    left: int
    top: int
    width: int
    height: int

    def to_mss_region(self) -> Dict[str, int]:
        """Convert to mss region format"""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height
        }


@dataclass
class FrameData:
    """Container for a captured frame with metadata"""
    timestamp: float
    frame_number: int
    regions: Dict[str, np.ndarray]  # region_name -> image array


class TripleBuffer:
    """
    Triple buffering system to prevent frame tearing and reduce latency.
    Uses three buffers: one being written to, one being read from, one ready to swap.
    """

    def __init__(self):
        self.buffers = [None, None, None]
        self.write_idx = 0
        self.read_idx = 1
        self.ready_idx = 2
        self.lock = threading.Lock()

    def write(self, data: FrameData):
        """Write new frame data to the write buffer"""
        with self.lock:
            self.buffers[self.write_idx] = data
            # Swap write and ready buffers
            self.write_idx, self.ready_idx = self.ready_idx, self.write_idx

    def read(self) -> Optional[FrameData]:
        """Read the most recent complete frame"""
        with self.lock:
            # If ready buffer has new data, swap it with read buffer
            if self.buffers[self.ready_idx] is not None:
                self.read_idx, self.ready_idx = self.ready_idx, self.read_idx
            return self.buffers[self.read_idx]


class ScreenCapture:
    """
    High-performance screen capture with ROI extraction.
    Captures at target FPS with minimal latency using triple buffering.
    """

    def __init__(
        self,
        target_fps: int = capture_cfg.TARGET_FPS,
        screen_width: int = capture_cfg.SCREEN_WIDTH,
        screen_height: int = capture_cfg.SCREEN_HEIGHT,
    ):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize MSS screen capture
        self.sct = mss.mss()

        # Setup capture regions
        self.regions = self._setup_regions()

        # Triple buffering
        self.buffer = TripleBuffer()

        # Capture thread control
        self.running = False
        self.capture_thread = None
        self.frame_count = 0

        # Performance metrics
        self.fps_actual = 0.0
        self.latency_ms = 0.0
        self.last_fps_update = time.time()
        self.fps_frame_count = 0

    def _setup_regions(self) -> Dict[str, CaptureRegion]:
        """Setup all ROI regions for capture"""
        regions = {}

        # Main viewport (center region for combat)
        main_left = (self.screen_width - capture_cfg.MAIN_VIEWPORT_WIDTH) // 2
        main_top = (self.screen_height - capture_cfg.MAIN_VIEWPORT_HEIGHT) // 2
        regions["main"] = CaptureRegion(
            name="main",
            left=main_left,
            top=main_top,
            width=capture_cfg.MAIN_VIEWPORT_WIDTH,
            height=capture_cfg.MAIN_VIEWPORT_HEIGHT
        )

        # Minimap (bottom-right corner)
        minimap_width = int(self.screen_width * capture_cfg.MINIMAP_WIDTH_RATIO)
        minimap_height = int(self.screen_height * capture_cfg.MINIMAP_HEIGHT_RATIO)
        minimap_left = self.screen_width - minimap_width - capture_cfg.MINIMAP_PADDING_RIGHT
        minimap_top = self.screen_height - minimap_height - capture_cfg.MINIMAP_PADDING_BOTTOM
        regions["minimap"] = CaptureRegion(
            name="minimap",
            left=minimap_left,
            top=minimap_top,
            width=minimap_width,
            height=minimap_height
        )

        # HUD (bottom center for health/mana/abilities)
        regions["hud"] = CaptureRegion(
            name="hud",
            left=capture_cfg.HUD_X_OFFSET,
            top=capture_cfg.HUD_Y_OFFSET,
            width=capture_cfg.HUD_WIDTH,
            height=capture_cfg.HUD_HEIGHT
        )

        return regions

    def capture_region(self, region: CaptureRegion) -> np.ndarray:
        """Capture a single screen region and return as numpy array"""
        screenshot = self.sct.grab(region.to_mss_region())

        # Convert to numpy array (mss returns BGRA)
        img = np.array(screenshot)

        # Convert BGRA to BGR (remove alpha channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def capture_all_regions(self) -> Dict[str, np.ndarray]:
        """Capture all defined regions"""
        captured = {}
        for name, region in self.regions.items():
            captured[name] = self.capture_region(region)
        return captured

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        # Re-initialize MSS in this thread (required for Linux/X11)
        self.sct = mss.mss()

        frame_num = 0
        last_frame_time = time.time()

        while self.running:
            loop_start = time.time()

            # Capture all regions
            regions = self.capture_all_regions()

            # Create frame data
            frame_data = FrameData(
                timestamp=time.time(),
                frame_number=frame_num,
                regions=regions
            )

            # Write to triple buffer
            self.buffer.write(frame_data)

            frame_num += 1
            self.fps_frame_count += 1

            # Calculate latency
            capture_time = time.time() - loop_start
            self.latency_ms = capture_time * 1000

            # Update FPS counter every second
            if time.time() - self.last_fps_update >= 1.0:
                self.fps_actual = self.fps_frame_count / (time.time() - self.last_fps_update)
                self.fps_frame_count = 0
                self.last_fps_update = time.time()

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start the capture thread"""
        if self.running:
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"Screen capture started at {self.target_fps} FPS")

    def stop(self):
        """Stop the capture thread"""
        if not self.running:
            return

        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        print("Screen capture stopped")

    def get_latest_frame(self) -> Optional[FrameData]:
        """Get the most recent captured frame"""
        return self.buffer.read()

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            "fps": self.fps_actual,
            "latency_ms": self.latency_ms,
            "target_fps": self.target_fps,
        }

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class RegionExtractor:
    """
    Utility class for extracting specific regions from captured frames.
    Useful for isolating areas for specific CV tasks.
    """

    @staticmethod
    def extract_ability_regions(hud_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract individual ability cooldown regions from HUD.
        Q, W, E, R ability boxes for OCR.
        """
        height, width = hud_image.shape[:2]

        # Ability positions (approximate, may need calibration)
        # These are normalized positions on the HUD
        ability_positions = {
            "Q": (0.15, 0.3, 0.08, 0.25),  # (x_start, x_end, y_start, y_end) normalized
            "W": (0.30, 0.45, 0.08, 0.25),
            "E": (0.45, 0.60, 0.08, 0.25),
            "R": (0.60, 0.75, 0.08, 0.25),
        }

        regions = {}
        for ability, (x1, x2, y1, y2) in ability_positions.items():
            x1_px = int(x1 * width)
            x2_px = int(x2 * width)
            y1_px = int(y1 * height)
            y2_px = int(y2 * height)
            regions[ability] = hud_image[y1_px:y2_px, x1_px:x2_px]

        return regions

    @staticmethod
    def extract_health_bar(hud_image: np.ndarray) -> np.ndarray:
        """Extract the health bar region for pixel-based HP detection"""
        height, width = hud_image.shape[:2]

        # Health bar position (normalized)
        x1 = int(0.20 * width)
        x2 = int(0.55 * width)
        y1 = int(0.60 * height)
        y2 = int(0.70 * height)

        return hud_image[y1:y2, x1:x2]

    @staticmethod
    def extract_mana_bar(hud_image: np.ndarray) -> np.ndarray:
        """Extract the mana bar region"""
        height, width = hud_image.shape[:2]

        # Mana bar position (normalized, below health bar)
        x1 = int(0.20 * width)
        x2 = int(0.55 * width)
        y1 = int(0.72 * height)
        y2 = int(0.80 * height)

        return hud_image[y1:y2, x1:x2]

    @staticmethod
    def calculate_bar_percentage(bar_image: np.ndarray, color_range: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Calculate the percentage filled of a colored bar (health/mana).

        Args:
            bar_image: The bar region image
            color_range: Tuple of (lower_bound, upper_bound) in HSV color space

        Returns:
            Percentage filled (0.0 to 1.0)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(bar_image, cv2.COLOR_BGR2HSV)

        # Create mask for the bar color
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)

        # Calculate percentage of colored pixels
        total_pixels = mask.size
        colored_pixels = cv2.countNonZero(mask)

        return colored_pixels / total_pixels if total_pixels > 0 else 0.0


# ============================================================================
# TESTING AND DEMO
# ============================================================================

if __name__ == "__main__":
    print("Testing Screen Capture Module")
    print("=" * 60)

    # Create capture instance
    capture = ScreenCapture(target_fps=60)

    # Start capturing
    capture.start()

    print("Capturing for 5 seconds...")
    print("Press Ctrl+C to stop early")

    try:
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 5.0:
            frame = capture.get_latest_frame()
            if frame is not None:
                frame_count += 1

                # Display one frame
                if frame_count % 60 == 0:  # Every second
                    stats = capture.get_performance_stats()
                    print(f"FPS: {stats['fps']:.1f} | Latency: {stats['latency_ms']:.2f}ms")

                    # Show minimap for visualization
                    if "minimap" in frame.regions:
                        cv2.imshow("Minimap", frame.regions["minimap"])
                        cv2.waitKey(1)

            time.sleep(0.001)  # Small sleep to prevent busy waiting

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        capture.stop()
        cv2.destroyAllWindows()

    print(f"\nCaptured {frame_count} frames")
    print("Final stats:", capture.get_performance_stats())
