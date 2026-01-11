"""
Simple overlay showing jungler visibility status.
"""
import tkinter as tk
from tkinter import font as tkfont
import threading
import time as time_module
from typing import Optional


class TrackerOverlay:
    """Simple overlay showing if jungler is visible."""

    COLORS = {
        'bg': '#1a1a2e',
        'fg': '#eaeaea',
        'visible': '#2ed573',
        'not_visible': '#6c6c8a',
        'alert': '#ff4757',
    }

    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Labels
        self._title_label: Optional[tk.Label] = None
        self._jungler_label: Optional[tk.Label] = None
        self._status_label: Optional[tk.Label] = None
        self._alert_label: Optional[tk.Label] = None

        # State
        self._jungler_name = ""
        self._status_text = "Waiting..."
        self._is_visible = False
        self._alert_text = ""
        self._alert_expire = 0

    def start(self):
        """Start overlay thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Run overlay."""
        self._root = tk.Tk()
        self._root.title("Jungler Tracker")
        self._root.attributes('-topmost', True)
        self._root.attributes('-alpha', 0.9)
        self._root.overrideredirect(True)
        self._root.geometry("+10+10")
        self._root.configure(bg=self.COLORS['bg'])

        frame = tk.Frame(self._root, bg=self.COLORS['bg'], padx=12, pady=8)
        frame.pack()

        # Title
        self._title_label = tk.Label(
            frame,
            text="JUNGLER TRACKER",
            font=("Segoe UI", 10, "bold"),
            fg="#4a9eff",
            bg=self.COLORS['bg']
        )
        self._title_label.pack(anchor='w')

        # Jungler name
        self._jungler_label = tk.Label(
            frame,
            text="Jungler: -",
            font=("Segoe UI", 9),
            fg=self.COLORS['fg'],
            bg=self.COLORS['bg']
        )
        self._jungler_label.pack(anchor='w', pady=(5, 0))

        # Status (VISIBLE / Not visible)
        self._status_label = tk.Label(
            frame,
            text="Waiting...",
            font=("Segoe UI", 12, "bold"),
            fg=self.COLORS['not_visible'],
            bg=self.COLORS['bg']
        )
        self._status_label.pack(anchor='w', pady=(5, 0))

        # Alert banner
        self._alert_label = tk.Label(
            frame,
            text="",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=self.COLORS['alert'],
            padx=8,
            pady=4
        )

        # Make draggable
        for widget in [frame, self._title_label, self._jungler_label, self._status_label]:
            widget.bind('<Button-1>', self._start_drag)
            widget.bind('<B1-Motion>', self._on_drag)

        self._update_loop()
        self._root.mainloop()

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _update_loop(self):
        """Update display."""
        if not self._running:
            return

        try:
            # Update jungler name
            if self._jungler_label:
                if self._jungler_name:
                    self._jungler_label.config(text=f"Jungler: {self._jungler_name}")
                else:
                    self._jungler_label.config(text="Jungler: -")

            # Update status
            if self._status_label:
                if self._is_visible:
                    self._status_label.config(
                        text="● VISIBLE",
                        fg=self.COLORS['visible']
                    )
                else:
                    self._status_label.config(
                        text=self._status_text,
                        fg=self.COLORS['not_visible']
                    )

            # Update alert
            if self._alert_label:
                now = time_module.time()
                if self._alert_text and now < self._alert_expire:
                    self._alert_label.config(text=self._alert_text)
                    self._alert_label.pack(anchor='w', pady=(8, 0), fill='x')
                else:
                    self._alert_label.pack_forget()
                    self._alert_text = ""

        except tk.TclError:
            return

        if self._root:
            self._root.after(100, self._update_loop)

    def update_jungler(self, name: str):
        """Set jungler name."""
        self._jungler_name = name

    def update_status(self, text: str):
        """Set status text."""
        self._status_text = text

    def set_visible(self, visible: bool):
        """Set if jungler is visible."""
        self._is_visible = visible

    def show_alert(self, text: str, duration: float = 3.0):
        """Show alert banner."""
        self._alert_text = text
        self._alert_expire = time_module.time() + duration

    def stop(self):
        """Stop overlay."""
        self._running = False
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except:
                pass


# Global instance
tracker_overlay = TrackerOverlay()
