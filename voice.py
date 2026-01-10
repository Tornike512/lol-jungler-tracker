"""
Text-to-speech alert system with cooldowns.
Uses pyttsx3 for offline voice synthesis.
"""
import time
import threading
import queue
from typing import Optional

import pyttsx3

from config import config
from zones import Zone, ThreatLevel


class VoiceAlertSystem:
    """Handles voice alerts with cooldown management."""

    def __init__(self):
        self._engine: Optional[pyttsx3.Engine] = None
        self._last_alert_time: float = 0
        self._alert_queue: queue.Queue = queue.Queue()
        self._voice_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._initialized: bool = False

    def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', config.settings.voice_rate)

            # Get available voices and use the first one
            voices = self._engine.getProperty('voices')
            if voices:
                self._engine.setProperty('voice', voices[0].id)

            self._initialized = True
            self._running = True

            # Start voice processing thread
            self._voice_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self._voice_thread.start()

            print("[STATUS] Voice system initialized")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize voice system: {e}")
            self._initialized = False
            return False

    def _process_alerts(self):
        """Background thread that processes voice alerts."""
        while self._running:
            try:
                # Wait for an alert with timeout
                message = self._alert_queue.get(timeout=0.5)
                if message and self._engine:
                    self._engine.say(message)
                    self._engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Voice alert failed: {e}")

    def _can_alert(self) -> bool:
        """Check if we're allowed to send an alert (cooldown check)."""
        if not config.settings.voice_enabled:
            return False

        current_time = time.time()
        time_since_last = current_time - self._last_alert_time
        return time_since_last >= config.settings.alert_cooldown_seconds

    def alert_disappeared(self, zone: Zone, champion: str) -> bool:
        """
        Alert that the jungler disappeared from a location.

        Args:
            zone: The zone where they were last seen
            champion: Champion name

        Returns:
            True if alert was triggered, False if suppressed by cooldown
        """
        if not self._can_alert():
            return False

        # Build the alert message
        threat_prefix = ""
        if zone.threat_level == ThreatLevel.DANGER:
            threat_prefix = "DANGER! "
        elif zone.threat_level == ThreatLevel.HIGH:
            threat_prefix = "Caution. "

        message = f"{threat_prefix}Enemy jungler disappeared from {zone.display_name}"

        return self._queue_alert(message)

    def alert_spotted(self, zone: Zone, champion: str) -> bool:
        """
        Alert that the jungler was spotted at a location.

        Args:
            zone: The zone where they were spotted
            champion: Champion name

        Returns:
            True if alert was triggered, False if suppressed by cooldown
        """
        if not self._can_alert():
            return False

        threat_prefix = ""
        if zone.threat_level == ThreatLevel.DANGER:
            threat_prefix = "DANGER! "

        message = f"{threat_prefix}Enemy jungler spotted at {zone.display_name}"

        return self._queue_alert(message)

    def alert_predicted(self, zone: Zone, confidence: float, champion: str) -> bool:
        """
        Alert with a predicted jungler location.

        Args:
            zone: The predicted zone
            confidence: Prediction confidence (0-1)
            champion: Champion name

        Returns:
            True if alert was triggered, False if suppressed by cooldown
        """
        if not self._can_alert():
            return False

        # Only announce predictions for dangerous zones
        if zone.threat_level.value < ThreatLevel.HIGH.value:
            return False

        # Adjust language based on confidence
        if confidence > 0.6:
            qualifier = "likely at"
        elif confidence > 0.4:
            qualifier = "possibly at"
        else:
            qualifier = "maybe near"

        threat_prefix = ""
        if zone.threat_level == ThreatLevel.DANGER:
            threat_prefix = "DANGER! "

        message = f"{threat_prefix}Enemy jungler {qualifier} {zone.display_name}"

        return self._queue_alert(message)

    def _queue_alert(self, message: str) -> bool:
        """Queue an alert message for speaking."""
        if not self._initialized:
            print(f"[ALERT] {message}")  # Fallback to console
            return True

        try:
            self._alert_queue.put_nowait(message)
            self._last_alert_time = time.time()
            print(f"[ALERT] {message}")
            return True
        except queue.Full:
            return False

    def get_cooldown_remaining(self) -> float:
        """Get seconds remaining on cooldown."""
        elapsed = time.time() - self._last_alert_time
        remaining = config.settings.alert_cooldown_seconds - elapsed
        return max(0, remaining)

    def is_on_cooldown(self) -> bool:
        """Check if voice is currently on cooldown."""
        return not self._can_alert()

    def shutdown(self):
        """Shutdown the voice system."""
        self._running = False
        if self._voice_thread and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=2)
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        print("[STATUS] Voice system shutdown")

    def reset_cooldown(self):
        """Reset the alert cooldown."""
        self._last_alert_time = 0


# Global voice system instance
voice_system = VoiceAlertSystem()
