"""
Text-to-speech alert system with cooldowns.
Uses pyttsx3 for offline voice synthesis.
"""
import time
from datetime import datetime
from typing import Optional

import pyttsx3

from config import config
from zones import Zone, ThreatLevel


def debug_log(message: str):
    """Print debug message with timestamp."""
    if config.settings.debug_mode:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG VOICE {timestamp}] {message}")


class VoiceAlertSystem:
    """Handles voice alerts with cooldown management."""

    def __init__(self):
        self._engine: Optional[pyttsx3.Engine] = None
        self._last_alert_time: float = 0
        self._initialized: bool = False

    def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            debug_log("Initializing TTS engine...")
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', config.settings.voice_rate)

            # Get available voices and use the first one
            voices = self._engine.getProperty('voices')
            debug_log(f"Available voices: {len(voices) if voices else 0}")
            if voices:
                self._engine.setProperty('voice', voices[0].id)
                debug_log(f"Using voice: {voices[0].name}")

            self._initialized = True
            debug_log("Voice system initialized successfully")
            print("[STATUS] Voice system initialized")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize voice system: {e}")
            debug_log(f"Init error: {e}")
            self._initialized = False
            return False

    def _speak(self, message: str):
        """Speak a message directly (blocking)."""
        if not self._initialized or not self._engine:
            debug_log("Engine not initialized, skipping speech")
            return

        try:
            debug_log(f"Speaking: {message}")
            self._engine.say(message)
            self._engine.runAndWait()
            debug_log("Speech completed")
        except Exception as e:
            debug_log(f"Speech error: {e}")
            # Try to reinitialize engine
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', config.settings.voice_rate)
                debug_log("Engine reinitialized after error")
            except:
                pass

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
            remaining = self.get_cooldown_remaining()
            debug_log(f"Alert BLOCKED by cooldown ({remaining:.1f}s remaining)")
            print(f"[VOICE] Alert blocked - cooldown {remaining:.1f}s remaining")
            return False

        # Build the alert message
        threat_prefix = ""
        if zone.threat_level == ThreatLevel.DANGER:
            threat_prefix = "DANGER! "
        elif zone.threat_level == ThreatLevel.HIGH:
            threat_prefix = "Caution. "

        message = f"{threat_prefix}Enemy jungler disappeared from {zone.display_name}"

        return self._trigger_alert(message)

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
            remaining = self.get_cooldown_remaining()
            debug_log(f"Spotted alert BLOCKED by cooldown ({remaining:.1f}s remaining)")
            print(f"[VOICE] Spotted alert blocked - cooldown {remaining:.1f}s remaining")
            return False

        threat_prefix = ""
        if zone.threat_level == ThreatLevel.DANGER:
            threat_prefix = "DANGER! "

        message = f"{threat_prefix}Enemy jungler spotted at {zone.display_name}"

        return self._trigger_alert(message)

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

        return self._trigger_alert(message)

    def _trigger_alert(self, message: str) -> bool:
        """Trigger an alert - speak it directly."""
        debug_log(f"Triggering alert: {message}")

        print(f"[ALERT] {message}")
        print(f"[VOICE] Speaking now...")
        self._last_alert_time = time.time()

        # Speak directly (blocking but short)
        self._speak(message)

        print(f"[VOICE] Speech completed, cooldown started ({config.settings.alert_cooldown_seconds}s)")

        return True

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
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        self._initialized = False
        print("[STATUS] Voice system shutdown")

    def reset_cooldown(self):
        """Reset the alert cooldown."""
        self._last_alert_time = 0


# Global voice system instance
voice_system = VoiceAlertSystem()
