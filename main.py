"""
League of Legends Jungler Tracker - Main Entry Point

A tool that tracks the enemy jungler on the minimap and provides voice alerts.
Designed for mid lane players to improve map awareness.

Usage:
    python main.py
"""
import sys
import time
import threading
import ctypes
from typing import Optional

# Windows-specific imports for game detection
try:
    import win32gui
    import win32process
    import psutil
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("[WARNING] pywin32 not installed. Game detection will be limited.")

from config import config
from capture import screen_capture
from detector import champion_detector, detect_enemy_jungler, is_game_active
from predictor import jungle_predictor
from voice import voice_system
from logger import detection_logger
from overlay import tracker_overlay
from zones import ThreatLevel


class JunglerTracker:
    """Main tracker application."""

    def __init__(self):
        self._running = False
        self._tracking = False
        self._game_active = False
        self._enemy_jungler = ""
        self._scan_thread: Optional[threading.Thread] = None
        self._last_scan_time = 0

    def start(self):
        """Start the tracker application."""
        print("=" * 60)
        print("  LEAGUE OF LEGENDS JUNGLER TRACKER")
        print("  For Mid Lane Players")
        print("=" * 60)
        print()

        self._running = True

        # Initialize systems
        print("[STATUS] Initializing systems...")

        # Initialize voice
        if config.settings.voice_enabled:
            voice_system.initialize()

        # Start overlay
        if config.settings.overlay_enabled:
            tracker_overlay.start()
            tracker_overlay.update_status("Waiting for game...")

        # Main loop - wait for game and track
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[STATUS] Interrupted by user")
        finally:
            self.stop()

    def _main_loop(self):
        """Main application loop."""
        while self._running:
            if not self._game_active:
                # Wait for game to start
                self._wait_for_game()

            if self._game_active and self._running:
                # Game found - start tracking
                self._setup_tracking()

                if self._tracking:
                    self._tracking_loop()

                # Game ended - cleanup
                self._end_tracking()

    def _wait_for_game(self):
        """Wait for League of Legends game to start."""
        print("[STATUS] Waiting for League of Legends game...")
        tracker_overlay.update_status("Waiting for game...")

        while self._running and not self._game_active:
            if self._detect_game():
                self._game_active = True
                print("[STATUS] Game detected!")
                break
            time.sleep(2)

    def _detect_game(self) -> bool:
        """Detect if a League of Legends game is running."""
        # First try Riot Live Client API - most reliable method
        if is_game_active():
            return True

        if not HAS_WIN32:
            # Fallback: try screen capture and look for minimap
            screen = screen_capture.capture_screen()
            if screen is not None:
                calibration = screen_capture.detect_minimap(screen)
                return calibration is not None
            return False

        # Check for game window
        try:
            def callback(hwnd, windows):
                title = win32gui.GetWindowText(hwnd)
                if "League of Legends" in title and "Client" in title:
                    windows.append(hwnd)
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                # Verify it's actually in-game (not client/lobby)
                for hwnd in windows:
                    if win32gui.IsWindowVisible(hwnd):
                        rect = win32gui.GetWindowRect(hwnd)
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        # In-game window is typically fullscreen or large
                        if width > 800 and height > 600:
                            return True
            return False
        except Exception:
            return False

    def _setup_tracking(self):
        """Set up tracking for a new game."""
        print("\n[STATUS] Setting up tracking...")
        tracker_overlay.update_status("Calibrating...")

        # Calibrate minimap
        time.sleep(2)  # Wait for game to fully load
        if not screen_capture.calibrate():
            print("[ERROR] Failed to calibrate minimap. Retrying...")
            time.sleep(5)
            if not screen_capture.calibrate():
                print("[ERROR] Calibration failed. Please ensure the game is visible.")
                self._game_active = False
                return

        # Get enemy jungler from user
        self._enemy_jungler = self._get_enemy_jungler()
        if not self._enemy_jungler:
            print("[STATUS] No jungler specified. Exiting tracking.")
            self._game_active = False
            return

        # Load champion template
        if not champion_detector.set_champion(self._enemy_jungler):
            print(f"[WARNING] Could not load icon for {self._enemy_jungler}. "
                  "Detection may be limited.")

        # Start logging
        detection_logger.start_game_session(self._enemy_jungler)

        # Update overlay
        tracker_overlay.update_jungler(self._enemy_jungler)
        tracker_overlay.update_status("Tracking active")

        print(f"\n[STATUS] Now tracking: {self._enemy_jungler}")
        print("[STATUS] Press Ctrl+C to stop\n")

        self._tracking = True
        config.state.is_running = True
        config.state.enemy_jungler = self._enemy_jungler

    def _get_enemy_jungler(self) -> str:
        """
        Get enemy jungler champion name.
        Tries auto-detection via Riot API first, falls back to manual input.
        """
        print("\n[STATUS] Attempting to auto-detect enemy jungler...")
        tracker_overlay.update_status("Detecting jungler...")

        # Try auto-detection via Riot Live Client API
        jungler = detect_enemy_jungler()

        if jungler:
            print(f"[SUCCESS] Auto-detected enemy jungler: {jungler}")
            # Confirm with user
            print(f"\nDetected jungler: {jungler}")
            try:
                confirm = input("[INPUT] Press Enter to confirm, or type a different champion: ").strip()
                if confirm:
                    return confirm
                return jungler
            except (EOFError, KeyboardInterrupt):
                return jungler
        else:
            # Fallback to manual input
            print("[INFO] Auto-detection failed (API unavailable or no Smite found)")
            print("\n" + "=" * 40)
            print("Enter the enemy jungler's champion name")
            print("(e.g., Lee Sin, Elise, Viego, etc.)")
            print("=" * 40)

            try:
                jungler = input("[INPUT] Enemy jungler: ").strip()
                return jungler
            except (EOFError, KeyboardInterrupt):
                return ""

    def _tracking_loop(self):
        """Main tracking loop - scans minimap and triggers alerts."""
        scan_interval = 1.0 / config.settings.scan_rate_hz
        last_visible = False
        last_zone_name = ""

        while self._running and self._tracking and self._game_active:
            loop_start = time.time()

            # Check if game is still running
            if not self._detect_game():
                print("[STATUS] Game ended or minimized.")
                break

            # Validate/recalibrate if needed
            if screen_capture.should_recalibrate():
                if not screen_capture.validate_calibration():
                    print("[STATUS] Recalibrating minimap...")
                    screen_capture.calibrate()

            # Capture minimap
            minimap = screen_capture.capture_minimap()
            if minimap is None:
                time.sleep(scan_interval)
                continue

            # Detect jungler
            detection = champion_detector.detect(minimap)
            current_time = time.time()

            if detection:
                zone = detection.zone
                zone_name = zone.name if zone else "unknown"
                zone_display = zone.display_name if zone else "unknown area"
                is_visible = detection.is_visible

                # Update predictor with confirmed sightings
                if is_visible and zone:
                    jungle_predictor.update_position(zone_name, current_time)
                    config.state.last_seen_position = detection.position
                    config.state.last_seen_time = current_time
                    config.state.last_seen_zone = zone_name
                    config.state.is_jungler_visible = True
                    config.state.prediction_active = False

                    # Update overlay
                    tracker_overlay.update_position(
                        zone_display, 0, "high", zone.threat_level, False
                    )

                # Check for visibility state change
                if champion_detector.just_disappeared() and zone:
                    # Jungler just went invisible - alert!
                    alert_triggered = voice_system.alert_disappeared(zone, self._enemy_jungler)

                    detection_logger.log_detection(
                        champion=self._enemy_jungler,
                        zone_name=zone_name,
                        zone_display=zone_display,
                        position=detection.position,
                        confidence="high",
                        is_visible=False,
                        alert_triggered=alert_triggered,
                        alert_suppressed_reason="cooldown" if not alert_triggered else ""
                    )

                    config.state.is_jungler_visible = False

                elif champion_detector.just_appeared() and zone:
                    # Jungler appeared - only alert for dangerous zones
                    alert_triggered = False
                    if zone.threat_level.value >= ThreatLevel.HIGH.value:
                        alert_triggered = voice_system.alert_spotted(zone, self._enemy_jungler)

                    detection_logger.log_detection(
                        champion=self._enemy_jungler,
                        zone_name=zone_name,
                        zone_display=zone_display,
                        position=detection.position,
                        confidence="high",
                        is_visible=True,
                        alert_triggered=alert_triggered
                    )

                # Prediction mode - jungler hasn't been seen for a while
                if not is_visible and config.state.last_seen_time:
                    time_since_seen = current_time - config.state.last_seen_time

                    if time_since_seen > config.settings.prediction_delay_seconds:
                        prediction = jungle_predictor.predict(current_time)

                        if prediction:
                            config.state.prediction_active = True
                            config.state.predicted_zone = prediction.zone.name

                            # Update overlay with prediction
                            confidence_str = f"{int(prediction.confidence * 100)}%"
                            tracker_overlay.update_position(
                                prediction.zone.display_name,
                                time_since_seen,
                                confidence_str,
                                prediction.zone.threat_level,
                                is_prediction=True
                            )

                            # Alert for dangerous predicted zones
                            if prediction.zone.threat_level == ThreatLevel.DANGER:
                                voice_system.alert_predicted(
                                    prediction.zone,
                                    prediction.confidence,
                                    self._enemy_jungler
                                )

                    elif config.state.last_seen_zone:
                        # Show last known position
                        last_zone = detection.zone
                        if last_zone:
                            tracker_overlay.update_position(
                                last_zone.display_name,
                                time_since_seen,
                                "last known",
                                last_zone.threat_level,
                                is_prediction=False
                            )

                last_visible = is_visible
                if zone:
                    last_zone_name = zone_name

            # Maintain scan rate
            elapsed = time.time() - loop_start
            if elapsed < scan_interval:
                time.sleep(scan_interval - elapsed)

    def _end_tracking(self):
        """Clean up after tracking ends."""
        print("\n[STATUS] Ending tracking session...")

        self._tracking = False
        self._game_active = False
        config.state.is_running = False

        # Save logs
        detection_logger.end_game_session()

        # Print stats
        stats = detection_logger.get_stats()
        if stats:
            print(f"\n[STATS] Session Summary:")
            print(f"  - Total detections: {stats.get('total_detections', 0)}")
            print(f"  - Alerts triggered: {stats.get('alerts_triggered', 0)}")
            print(f"  - Game duration: {int(stats.get('game_duration', 0) / 60)}m")

        # Reset systems
        champion_detector.reset()
        jungle_predictor.reset()
        config.reset_state()

        tracker_overlay.clear_position()
        tracker_overlay.update_status("Game ended")
        tracker_overlay.update_jungler("")

        print("[STATUS] Ready for next game...\n")

    def stop(self):
        """Stop the tracker."""
        print("\n[STATUS] Shutting down...")

        self._running = False
        self._tracking = False

        # Shutdown systems
        voice_system.shutdown()
        tracker_overlay.stop()

        if detection_logger._current_log_file:
            detection_logger.end_game_session()

        print("[STATUS] Goodbye!")


def main():
    """Main entry point."""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    # Try to enable DPI awareness on Windows
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    # Create and start tracker
    tracker = JunglerTracker()

    try:
        tracker.start()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        tracker.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
