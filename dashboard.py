"""
Real-time Dashboard for LoL RL Agent
Displays detections, Q-values, and performance metrics during training/execution.
"""
import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Optional, Dict, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

from src.config import logging_cfg


class Dashboard:
    """
    Real-time dashboard showing agent state and performance.
    """

    def __init__(self, update_freq_ms: int = logging_cfg.DASHBOARD_UPDATE_FREQ_MS):
        self.update_freq_ms = update_freq_ms
        self.running = False

        # Data storage
        self.current_state: Optional[Dict[str, Any]] = None
        self.reward_history = []
        self.fps_history = []
        self.apm_history = []
        self.max_history_length = 100

        # Create GUI
        self.root = tk.Tk()
        self.root.title("LoL RL Agent Dashboard")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._create_widgets()

    def _create_widgets(self):
        """Create dashboard widgets"""
        # Top frame for current state
        state_frame = ttk.LabelFrame(self.root, text="Current State", padding=10)
        state_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # State labels
        self.hp_label = ttk.Label(state_frame, text="HP: --", font=("Arial", 12))
        self.hp_label.grid(row=0, column=0, padx=10)

        self.mana_label = ttk.Label(state_frame, text="Mana: --", font=("Arial", 12))
        self.mana_label.grid(row=0, column=1, padx=10)

        self.detections_label = ttk.Label(state_frame, text="Detections: --", font=("Arial", 12))
        self.detections_label.grid(row=0, column=2, padx=10)

        self.apm_label = ttk.Label(state_frame, text="APM: --", font=("Arial", 12))
        self.apm_label.grid(row=0, column=3, padx=10)

        # Performance frame
        perf_frame = ttk.LabelFrame(self.root, text="Performance", padding=10)
        perf_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        self.fps_label = ttk.Label(perf_frame, text="FPS: --", font=("Arial", 12))
        self.fps_label.grid(row=0, column=0, padx=10)

        self.latency_label = ttk.Label(perf_frame, text="Latency: --", font=("Arial", 12))
        self.latency_label.grid(row=0, column=1, padx=10)

        self.reward_label = ttk.Label(perf_frame, text="Episode Reward: --", font=("Arial", 12))
        self.reward_label.grid(row=0, column=2, padx=10)

        # Plot frame
        if MATPLOTLIB_AVAILABLE:
            plot_frame = ttk.Frame(self.root)
            plot_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)

            # Create matplotlib figure
            self.fig = Figure(figsize=(12, 6))

            # Reward plot
            self.ax_reward = self.fig.add_subplot(131)
            self.ax_reward.set_title("Episode Reward")
            self.ax_reward.set_xlabel("Episode")
            self.ax_reward.set_ylabel("Reward")
            self.line_reward, = self.ax_reward.plot([], [], 'b-')

            # FPS plot
            self.ax_fps = self.fig.add_subplot(132)
            self.ax_fps.set_title("FPS")
            self.ax_fps.set_xlabel("Time")
            self.ax_fps.set_ylabel("Frames/sec")
            self.line_fps, = self.ax_fps.plot([], [], 'g-')

            # APM plot
            self.ax_apm = self.fig.add_subplot(133)
            self.ax_apm.set_title("APM")
            self.ax_apm.set_xlabel("Time")
            self.ax_apm.set_ylabel("Actions/min")
            self.line_apm, = self.ax_apm.plot([], [], 'r-')

            self.fig.tight_layout()

            # Embed in tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def update_state(self, state: Dict[str, Any]):
        """Update dashboard with new state"""
        self.current_state = state

        # Update reward history
        if "episode_reward" in state:
            self.reward_history.append(state["episode_reward"])
            if len(self.reward_history) > self.max_history_length:
                self.reward_history.pop(0)

        # Update FPS history
        if "fps" in state:
            self.fps_history.append(state["fps"])
            if len(self.fps_history) > self.max_history_length:
                self.fps_history.pop(0)

        # Update APM history
        if "apm" in state:
            self.apm_history.append(state["apm"])
            if len(self.apm_history) > self.max_history_length:
                self.apm_history.pop(0)

    def _update_gui(self):
        """Update GUI elements (called periodically)"""
        if not self.running:
            return

        if self.current_state is not None:
            # Update state labels
            hp = self.current_state.get("player_hp", 0.0)
            self.hp_label.config(text=f"HP: {hp:.1%}")

            mana = self.current_state.get("player_mana", 0.0)
            self.mana_label.config(text=f"Mana: {mana:.1%}")

            detections = self.current_state.get("detections", 0)
            self.detections_label.config(text=f"Detections: {detections}")

            apm = self.current_state.get("apm", 0.0)
            self.apm_label.config(text=f"APM: {apm:.1f}")

            # Update performance labels
            fps = self.current_state.get("fps", 0.0)
            self.fps_label.config(text=f"FPS: {fps:.1f}")

            latency = self.current_state.get("latency_ms", 0.0)
            self.latency_label.config(text=f"Latency: {latency:.2f}ms")

            reward = self.current_state.get("episode_reward", 0.0)
            self.reward_label.config(text=f"Episode Reward: {reward:.2f}")

            # Update plots
            if MATPLOTLIB_AVAILABLE:
                self._update_plots()

        # Schedule next update
        self.root.after(self.update_freq_ms, self._update_gui)

    def _update_plots(self):
        """Update matplotlib plots"""
        # Update reward plot
        if len(self.reward_history) > 0:
            self.line_reward.set_data(range(len(self.reward_history)), self.reward_history)
            self.ax_reward.relim()
            self.ax_reward.autoscale_view()

        # Update FPS plot
        if len(self.fps_history) > 0:
            self.line_fps.set_data(range(len(self.fps_history)), self.fps_history)
            self.ax_fps.relim()
            self.ax_fps.autoscale_view()

        # Update APM plot
        if len(self.apm_history) > 0:
            self.line_apm.set_data(range(len(self.apm_history)), self.apm_history)
            self.ax_apm.relim()
            self.ax_apm.autoscale_view()

        self.canvas.draw()

    def start(self):
        """Start the dashboard"""
        self.running = True
        self._update_gui()

        # Run in main thread
        self.root.mainloop()

    def start_async(self):
        """Start the dashboard in a separate thread"""
        self.running = True
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def close(self):
        """Close the dashboard"""
        self.running = False
        self.root.quit()
        self.root.destroy()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Dashboard")
    print("=" * 60)

    # Create dashboard
    dashboard = Dashboard()

    # Simulate updates in background
    def simulate_updates():
        episode_reward = 0.0
        for i in range(100):
            time.sleep(0.1)

            episode_reward += np.random.randn() * 0.5

            # Simulate state
            state = {
                "player_hp": 0.5 + 0.3 * np.sin(i * 0.1),
                "player_mana": 0.7 + 0.2 * np.cos(i * 0.15),
                "detections": int(5 + 3 * np.sin(i * 0.2)),
                "apm": 200 + 50 * np.sin(i * 0.3),
                "fps": 55 + 5 * np.random.randn(),
                "latency_ms": 10 + 2 * np.random.randn(),
                "episode_reward": episode_reward,
            }

            dashboard.update_state(state)

    # Start simulation in background
    update_thread = threading.Thread(target=simulate_updates, daemon=True)
    update_thread.start()

    # Start dashboard (blocks until closed)
    dashboard.start()

    print("Dashboard closed")
