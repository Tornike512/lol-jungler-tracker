"""
On-screen overlay for displaying tracker status.
Uses tkinter for a lightweight, always-on-top overlay window.
"""
import tkinter as tk
from tkinter import font as tkfont
import threading
import time
from typing import Optional, Callable

from config import config
from zones import ThreatLevel


class TrackerOverlay:
    """Non-intrusive overlay window showing tracker status."""

    # Colors
    COLORS = {
        'bg': '#1a1a2e',
        'fg': '#eaeaea',
        'accent': '#4a9eff',
        'danger': '#ff4757',
        'warning': '#ffa502',
        'safe': '#2ed573',
        'dim': '#6c6c8a'
    }

    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._visible = True

        # UI elements
        self._status_label: Optional[tk.Label] = None
        self._jungler_label: Optional[tk.Label] = None
        self._position_label: Optional[tk.Label] = None
        self._time_label: Optional[tk.Label] = None
        self._confidence_label: Optional[tk.Label] = None

        # State
        self._current_status = "Initializing..."
        self._current_jungler = ""
        self._current_position = "Unknown"
        self._current_time = ""
        self._current_confidence = ""
        self._threat_level = ThreatLevel.NONE

        # Toggle callback
        self._toggle_callback: Optional[Callable] = None

    def start(self):
        """Start the overlay in a separate thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_overlay, daemon=True)
        self._thread.start()

    def _run_overlay(self):
        """Main overlay thread function."""
        self._root = tk.Tk()
        self._root.title("LoL Jungler Tracker")

        # Configure window
        self._root.attributes('-topmost', True)
        self._root.attributes('-alpha', config.settings.overlay_opacity)
        self._root.overrideredirect(True)  # Remove window decorations

        # Position in top-left corner
        x, y = config.settings.overlay_position
        self._root.geometry(f"+{x}+{y}")

        # Set background
        self._root.configure(bg=self.COLORS['bg'])

        # Create frame with padding
        frame = tk.Frame(self._root, bg=self.COLORS['bg'], padx=10, pady=8)
        frame.pack()

        # Title
        title_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        title = tk.Label(
            frame,
            text="JUNGLER TRACKER",
            font=title_font,
            fg=self.COLORS['accent'],
            bg=self.COLORS['bg']
        )
        title.pack(anchor='w')

        # Status
        self._status_label = tk.Label(
            frame,
            text="Status: Initializing...",
            font=("Segoe UI", 9),
            fg=self.COLORS['fg'],
            bg=self.COLORS['bg']
        )
        self._status_label.pack(anchor='w', pady=(5, 0))

        # Jungler name
        self._jungler_label = tk.Label(
            frame,
            text="Tracking: -",
            font=("Segoe UI", 9),
            fg=self.COLORS['dim'],
            bg=self.COLORS['bg']
        )
        self._jungler_label.pack(anchor='w')

        # Separator
        sep = tk.Frame(frame, height=1, bg=self.COLORS['dim'])
        sep.pack(fill='x', pady=5)

        # Position
        self._position_label = tk.Label(
            frame,
            text="Position: Unknown",
            font=("Segoe UI", 10, "bold"),
            fg=self.COLORS['fg'],
            bg=self.COLORS['bg']
        )
        self._position_label.pack(anchor='w')

        # Time since seen
        self._time_label = tk.Label(
            frame,
            text="Last seen: -",
            font=("Segoe UI", 9),
            fg=self.COLORS['dim'],
            bg=self.COLORS['bg']
        )
        self._time_label.pack(anchor='w')

        # Confidence
        self._confidence_label = tk.Label(
            frame,
            text="",
            font=("Segoe UI", 8),
            fg=self.COLORS['dim'],
            bg=self.COLORS['bg']
        )
        self._confidence_label.pack(anchor='w')

        # Make window draggable
        self._make_draggable(frame)
        self._make_draggable(title)

        # Bind right-click to toggle visibility
        self._root.bind('<Button-3>', self._on_right_click)

        # Start update loop
        self._update_loop()

        # Run tkinter main loop
        try:
            self._root.mainloop()
        except Exception:
            pass

    def _make_draggable(self, widget):
        """Make a widget draggable."""
        widget.bind('<Button-1>', self._start_drag)
        widget.bind('<B1-Motion>', self._on_drag)

    def _start_drag(self, event):
        """Start dragging the window."""
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        """Handle window dragging."""
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _on_right_click(self, event):
        """Handle right-click to minimize/restore."""
        if self._toggle_callback:
            self._toggle_callback()

    def _update_loop(self):
        """Periodic update of overlay content."""
        if not self._running:
            return

        try:
            # Update status
            if self._status_label:
                self._status_label.config(text=f"Status: {self._current_status}")

            # Update jungler
            if self._jungler_label:
                if self._current_jungler:
                    self._jungler_label.config(text=f"Tracking: {self._current_jungler}")
                else:
                    self._jungler_label.config(text="Tracking: -")

            # Update position with threat-based color
            if self._position_label:
                color = self._get_threat_color()
                self._position_label.config(
                    text=f"Position: {self._current_position}",
                    fg=color
                )

            # Update time
            if self._time_label:
                self._time_label.config(text=f"Last seen: {self._current_time}")

            # Update confidence
            if self._confidence_label:
                self._confidence_label.config(text=self._current_confidence)

        except tk.TclError:
            # Window was destroyed
            return

        # Schedule next update
        if self._root:
            self._root.after(100, self._update_loop)

    def _get_threat_color(self) -> str:
        """Get color based on current threat level."""
        if self._threat_level == ThreatLevel.DANGER:
            return self.COLORS['danger']
        elif self._threat_level == ThreatLevel.HIGH:
            return self.COLORS['warning']
        elif self._threat_level in (ThreatLevel.MEDIUM, ThreatLevel.LOW):
            return self.COLORS['safe']
        return self.COLORS['fg']

    def update_status(self, status: str):
        """Update the tracker status text."""
        self._current_status = status

    def update_jungler(self, champion: str):
        """Update the tracked jungler name."""
        self._current_jungler = champion

    def update_position(
        self,
        zone_name: str,
        time_since_seen: float,
        confidence: str,
        threat_level: ThreatLevel,
        is_prediction: bool = False
    ):
        """Update position display."""
        # Format position text
        if is_prediction:
            self._current_position = f"~{zone_name}"
            self._current_confidence = f"Confidence: {confidence} (predicted)"
        else:
            self._current_position = zone_name
            self._current_confidence = f"Confidence: {confidence}"

        # Format time
        if time_since_seen < 1:
            self._current_time = "Just now"
        elif time_since_seen < 60:
            self._current_time = f"{int(time_since_seen)}s ago"
        else:
            minutes = int(time_since_seen // 60)
            seconds = int(time_since_seen % 60)
            self._current_time = f"{minutes}m {seconds}s ago"

        self._threat_level = threat_level

    def clear_position(self):
        """Clear position display (jungler not tracked)."""
        self._current_position = "Unknown"
        self._current_time = "-"
        self._current_confidence = ""
        self._threat_level = ThreatLevel.NONE

    def set_toggle_callback(self, callback: Callable):
        """Set callback for when overlay is toggled."""
        self._toggle_callback = callback

    def toggle_visibility(self):
        """Toggle overlay visibility."""
        self._visible = not self._visible
        if self._root:
            try:
                if self._visible:
                    self._root.deiconify()
                else:
                    self._root.withdraw()
            except tk.TclError:
                pass

    def stop(self):
        """Stop the overlay."""
        self._running = False
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
        self._root = None


# Global overlay instance
tracker_overlay = TrackerOverlay()
