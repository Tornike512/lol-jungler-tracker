"""
Detection logging system.
Saves all detections to CSV/JSON files for later analysis.
"""
import os
import json
import csv
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, asdict

from config import LOGS_DIR


@dataclass
class LogEntry:
    """A single log entry for a detection event."""
    timestamp: str
    game_time: float  # Seconds since game start (if available)
    champion: str
    zone: str
    zone_display: str
    position_x: float
    position_y: float
    confidence: str  # "high", "medium", "low", "prediction"
    is_visible: bool
    alert_triggered: bool
    alert_suppressed_reason: str  # "cooldown", "low_priority", ""
    prediction_zone: str  # If in prediction mode


class DetectionLogger:
    """Handles logging of all detection events."""

    def __init__(self):
        self._current_log_file: Optional[str] = None
        self._entries: List[LogEntry] = []
        self._game_start_time: Optional[datetime] = None
        self._csv_writer = None
        self._csv_file = None

    def start_game_session(self, champion: str):
        """Start a new logging session for a game."""
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_log_file = os.path.join(
            LOGS_DIR, f"tracking_{champion}_{timestamp}"
        )
        self._entries = []
        self._game_start_time = datetime.now()

        # Initialize CSV file
        csv_path = f"{self._current_log_file}.csv"
        self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)

        # Write header
        self._csv_writer.writerow([
            'timestamp', 'game_time', 'champion', 'zone', 'zone_display',
            'position_x', 'position_y', 'confidence', 'is_visible',
            'alert_triggered', 'alert_suppressed_reason', 'prediction_zone'
        ])

        print(f"[LOG] Started logging to {csv_path}")

    def log_detection(
        self,
        champion: str,
        zone_name: str,
        zone_display: str,
        position: tuple,
        confidence: str,
        is_visible: bool,
        alert_triggered: bool,
        alert_suppressed_reason: str = "",
        prediction_zone: str = ""
    ):
        """Log a detection event."""
        now = datetime.now()
        game_time = 0.0
        if self._game_start_time:
            game_time = (now - self._game_start_time).total_seconds()

        entry = LogEntry(
            timestamp=now.strftime("%H:%M:%S.%f")[:-3],
            game_time=round(game_time, 1),
            champion=champion,
            zone=zone_name,
            zone_display=zone_display,
            position_x=round(position[0], 3) if position else 0,
            position_y=round(position[1], 3) if position else 0,
            confidence=confidence,
            is_visible=is_visible,
            alert_triggered=alert_triggered,
            alert_suppressed_reason=alert_suppressed_reason,
            prediction_zone=prediction_zone
        )

        self._entries.append(entry)

        # Write to CSV immediately
        if self._csv_writer:
            self._csv_writer.writerow([
                entry.timestamp,
                entry.game_time,
                entry.champion,
                entry.zone,
                entry.zone_display,
                entry.position_x,
                entry.position_y,
                entry.confidence,
                entry.is_visible,
                entry.alert_triggered,
                entry.alert_suppressed_reason,
                entry.prediction_zone
            ])
            self._csv_file.flush()

        # Console output for significant events
        status = "visible" if is_visible else "invisible"
        alert_status = "announced" if alert_triggered else "suppressed" if alert_suppressed_reason else ""

        if alert_triggered or (not is_visible and zone_name):
            print(f"[LOG] {entry.timestamp} | {champion} | {zone_name} | "
                  f"{confidence} | {alert_status}")

    def end_game_session(self):
        """End the current logging session and save final files."""
        if not self._current_log_file:
            return

        # Close CSV
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # Save JSON summary
        json_path = f"{self._current_log_file}.json"
        summary = {
            "game_start": self._game_start_time.isoformat() if self._game_start_time else None,
            "game_end": datetime.now().isoformat(),
            "total_detections": len(self._entries),
            "alerts_triggered": sum(1 for e in self._entries if e.alert_triggered),
            "entries": [asdict(e) for e in self._entries]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"[LOG] Session ended. Saved {len(self._entries)} entries to {self._current_log_file}")

        # Reset state
        self._current_log_file = None
        self._entries = []
        self._game_start_time = None

    def get_recent_entries(self, count: int = 10) -> List[LogEntry]:
        """Get the most recent log entries."""
        return self._entries[-count:] if self._entries else []

    def get_stats(self) -> dict:
        """Get statistics for the current session."""
        if not self._entries:
            return {}

        visible_count = sum(1 for e in self._entries if e.is_visible)
        alert_count = sum(1 for e in self._entries if e.alert_triggered)
        suppressed_count = sum(1 for e in self._entries if e.alert_suppressed_reason)

        return {
            "total_detections": len(self._entries),
            "visible_detections": visible_count,
            "invisible_detections": len(self._entries) - visible_count,
            "alerts_triggered": alert_count,
            "alerts_suppressed": suppressed_count,
            "game_duration": (datetime.now() - self._game_start_time).total_seconds()
            if self._game_start_time else 0
        }


# Global logger instance
detection_logger = DetectionLogger()
