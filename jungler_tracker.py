#!/usr/bin/env python3
"""
League of Legends Predictive Jungler Pathing Helper
====================================================
A legal, manual-input overlay for tracking enemy jungle pathing.
Requires you to press hotkeys when you witness camps being taken.

Features:
- F1-F6 hotkeys to log camp sightings (Blue, Red, Gromp, Krugs, Raptors, Wolves)
- Predicts likely pathing based on respawn timers
- Shows "Danger Level" based on theoretical position
- Manual input only - no automation

Usage:
- Run while in-game
- When you SEE enemy jungler take a camp, press corresponding F-key
- Overlay updates predictions
"""

import tkinter as tk
import requests
import urllib3
import time
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Disable SSL warnings for local API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Riot Live Client API
API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_INTERVAL_MS = 1000  # 1 second updates

# Camp definitions: (Name, Respawn time in seconds, Position weight)
CAMPS = {
    'F1': ("Blue Buff", 300, "Blue Side"),      # 5 minutes
    'F2': ("Red Buff", 300, "Red Side"),        # 5 minutes  
    'F3': ("Gromp", 150, "Blue Side"),          # 2.5 minutes
    'F4': ("Krugs", 150, "Red Side"),           # 2.5 minutes
    'F5': ("Raptors", 150, "Mid Side"),         # 2.5 minutes
    'F6': ("Wolves", 150, "Mid Side"),          # 2.5 minutes
}

COLORS = {
    'safe': '#00FF00',      # Green - likely far away
    'caution': '#FFFF00',   # Yellow - could be approaching
    'danger': '#FF0000',    # Red - likely nearby/ganking
    'unknown': '#808080',   # Gray - no data
    'bg': '#1a1a1a',        # Dark background
    'text': '#ffffff'       # White text
}

@dataclass
class CampTimer:
    name: str
    respawn_at: float
    side: str

class JunglerPathingHelper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Jungle Pathing Helper")
        self.root.geometry("350x400")
        self.root.configure(bg=COLORS['bg'])
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        
        # Position top-right
        screen_width = self.root.winfo_screenwidth()
        self.root.geometry(f"+{screen_width - 370}+50")
        
        self.camp_timers: Dict[str, CampTimer] = {}
        self.enemy_jungler = None
        self.my_position = "BOT"  # Assume bot lane, user can change
        self.last_prediction = "Waiting for data..."
        
        self.setup_ui()
        self.setup_hotkeys()
        
        # Start API polling
        self.update_game_data()
        
    def setup_ui(self):
        """Create the overlay interface."""
        # Header
        header = tk.Label(
            self.root, 
            text="🎯 Jungler Pathing Helper", 
            font=('Helvetica', 14, 'bold'),
            bg=COLORS['bg'], 
            fg=COLORS['text']
        )
        header.pack(pady=10)
        
        # Instructions
        instr = tk.Label(
            self.root,
            text="Press F1-F6 when you SEE camps taken\nF1=Blue F2=Red F3=Gromp F4=Krugs F5=Raptors F6=Wolves",
            font=('Helvetica', 9),
            bg=COLORS['bg'],
            fg='#aaaaaa',
            justify=tk.LEFT
        )
        instr.pack(pady=5)
        
        # Status Frame
        self.status_frame = tk.Frame(self.root, bg=COLORS['bg'])
        self.status_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Status: Unknown",
            font=('Helvetica', 12, 'bold'),
            bg=COLORS['unknown'],
            fg='white',
            width=20,
            height=2
        )
        self.status_label.pack()
        
        # Prediction Text
        self.prediction_label = tk.Label(
            self.root,
            text="No camp data yet.\nWatch enemy jungler and log camps!",
            font=('Helvetica', 10),
            bg=COLORS['bg'],
            fg=COLORS['text'],
            wraplength=300,
            justify=tk.LEFT
        )
        self.prediction_label.pack(pady=10)
        
        # Camp Timers List
        self.timers_text = tk.Text(
            self.root,
            height=10,
            width=40,
            bg='#2a2a2a',
            fg=COLORS['text'],
            font=('Courier', 9),
            state=tk.DISABLED
        )
        self.timers_text.pack(pady=10, padx=10)
        
        # Lane selector
        lane_frame = tk.Frame(self.root, bg=COLORS['bg'])
        lane_frame.pack(pady=5)
        tk.Label(lane_frame, text="Your Lane:", bg=COLORS['bg'], fg='white').pack(side=tk.LEFT)
        self.lane_var = tk.StringVar(value="BOT")
        for lane in ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]:
            tk.Radiobutton(
                lane_frame, 
                text=lane, 
                variable=self.lane_var, 
                value=lane,
                bg=COLORS['bg'],
                fg='white',
                selectcolor=COLORS['bg'],
                command=self.update_prediction
            ).pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg=COLORS['bg'])
        btn_frame.pack(pady=5)
        
        tk.Button(
            btn_frame,
            text="Clear All",
            command=self.clear_timers,
            bg='#444444',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Exit",
            command=self.root.quit,
            bg='#444444',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
    def setup_hotkeys(self):
        """Set up global hotkeys for camp logging."""
        for key in CAMPS.keys():
            self.root.bind(f'<{key}>', lambda e, k=key: self.log_camp(k))
            
    def log_camp(self, key: str):
        """Log that you saw the enemy take a camp."""
        camp_name, respawn, side = CAMPS[key]
        current_time = time.time()
        respawn_time = current_time + respawn
        
        self.camp_timers[camp_name] = CampTimer(camp_name, respawn_time, side)
        self.update_prediction()
        self.update_timers_display()
        
        # Visual feedback
        self.flash_status(f"Logged: {camp_name}", COLORS['safe'])
        
    def flash_status(self, text, color):
        """Flash a temporary status message."""
        original_text = self.status_label.cget("text")
        original_bg = self.status_label.cget("bg")
        
        self.status_label.config(text=text, bg=color)
        self.root.after(1000, lambda: self.status_label.config(text=original_text, bg=original_bg))
        
    def clear_timers(self):
        """Clear all camp timers."""
        self.camp_timers.clear()
        self.update_prediction()
        self.update_timers_display()
        
    def get_game_data(self) -> Optional[Dict]:
        """Fetch game data from Live Client API."""
        try:
            response = requests.get(API_URL, timeout=2, verify=False)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
        
    def update_game_data(self):
        """Poll game data to identify enemy jungler."""
        data = self.get_game_data()
        if data:
            # Find enemy jungler
            my_team = None
            for player in data.get('allPlayers', []):
                if player.get('summonerName') == data.get('activePlayer', {}).get('summonerName'):
                    my_team = player.get('team')
                    break
                    
            if my_team:
                for player in data.get('allPlayers', []):
                    if player.get('team') != my_team and player.get('position') == 'JUNGLE':
                        if self.enemy_jungler != player.get('championName'):
                            self.enemy_jungler = player.get('championName')
                            self.prediction_label.config(
                                text=f"Tracking: {self.enemy_jungler}\nWaiting for camp data..."
                            )
        
        # Schedule next update
        self.root.after(POLL_INTERVAL_MS, self.update_game_data)
        
    def analyze_pathing(self) -> Tuple[str, str, str]:
        """
        Analyze camp timers to predict pathing.
        Returns: (danger_level, prediction_text, color)
        """
        if not self.camp_timers:
            return "unknown", "No data. Watch enemy jungler and press F1-F6 when you see camps taken.", COLORS['unknown']
            
        current_time = time.time()
        my_lane = self.lane_var.get()
        
        # Find upcoming respawns (next 60 seconds)
        upcoming = []
        recently_taken = []  # Taken in last 30 seconds
        
        for camp_name, timer in self.camp_timers.items():
            time_until = timer.respawn_at - current_time
            time_since_taken = current_time - (timer.respawn_at - self.get_camp_respawn(camp_name))
            
            if 0 < time_until < 60:
                upcoming.append((camp_name, time_until, timer.side))
            elif time_since_taken < 30:
                recently_taken.append((camp_name, timer.side))
                
        # Logic for predictions
        danger_level = "safe"
        prediction = ""
        
        # If camps were taken recently on our side -> DANGER
        for camp, side in recently_taken:
            if self.is_my_side(side, my_lane):
                danger_level = "danger"
                prediction = f"⚠️ DANGER: Just took {camp} on {side}! Likely ganking {my_lane} now or clearing toward you."
                break
                
        # If camps respawning soon on our side -> CAUTION
        if danger_level == "safe" and upcoming:
            for camp, time_left, side in upcoming:
                if self.is_my_side(side, my_lane):
                    danger_level = "caution"
                    prediction = f"⚡ CAUTION: {camp} respawns in {int(time_left)}s on {side}. May path here soon."
                    break
                    
        # If no threats detected
        if danger_level == "safe":
            if upcoming:
                camps_str = ", ".join([f"{c} ({int(t)}s)" for c, t, s in upcoming[:2]])
                prediction = f"✅ SAFE: Enemy likely farming {camps_str}. Good time to trade/push."
            elif recently_taken:
                last_camp = recently_taken[-1][0]
                prediction = f"✅ SAFE: Last seen at {last_camp}. Likely on opposite side or backed."
            else:
                prediction = "No immediate threats detected. Enemy jungle location unknown."
                
        return danger_level, prediction, COLORS[danger_level]
        
    def get_camp_respawn(self, camp_name: str) -> int:
        """Get respawn time for a camp."""
        for key, (name, respawn, side) in CAMPS.items():
            if name == camp_name:
                return respawn
        return 150
        
    def is_my_side(self, camp_side: str, my_lane: str) -> bool:
        """Determine if a camp side is dangerous for my lane."""
        lane_sides = {
            "TOP": ["Blue Side", "Mid Side"],      # Blue buff/Gromp side
            "MID": ["Mid Side"],                    # Center
            "BOT": ["Red Side", "Mid Side"],       # Red buff/Krugs side
            "SUPPORT": ["Red Side", "Mid Side"],
            "JUNGLE": ["Blue Side", "Red Side", "Mid Side"]
        }
        return camp_side in lane_sides.get(my_lane, [])
        
    def update_prediction(self):
        """Update the prediction display."""
        danger, prediction, color = self.analyze_pathing()
        self.last_prediction = prediction
        
        self.status_label.config(
            text=f"Status: {danger.upper()}",
            bg=color
        )
        self.prediction_label.config(text=prediction)
        self.update_timers_display()
        
    def update_timers_display(self):
        """Update the camp timers list."""
        self.timers_text.config(state=tk.NORMAL)
        self.timers_text.delete(1.0, tk.END)
        
        current_time = time.time()
        
        if not self.camp_timers:
            self.timers_text.insert(tk.END, "No camps logged yet.\n")
            self.timers_text.insert(tk.END, "Press F1-F6 when you see camps taken.\n\n")
            self.timers_text.insert(tk.END, "F1: Blue Buff    F2: Red Buff\n")
            self.timers_text.insert(tk.END, "F3: Gromp        F4: Krugs\n")
            self.timers_text.insert(tk.END, "F5: Raptors      F6: Wolves\n")
        else:
            # Sort by respawn time
            sorted_camps = sorted(
                self.camp_timers.items(),
                key=lambda x: x[1].respawn_at
            )
            
            self.timers_text.insert(tk.END, "Camp Timers:\n")
            self.timers_text.insert(tk.END, "-" * 30 + "\n")
            
            for camp_name, timer in sorted_camps:
                time_left = timer.respawn_at - current_time
                if time_left > 0:
                    minutes = int(time_left // 60)
                    seconds = int(time_left % 60)
                    status = f"{minutes}:{seconds:02d}"
                else:
                    status = "READY"
                    
                side_short = timer.side.replace(" Side", "")[0]
                self.timers_text.insert(tk.END, f"{camp_name:<12} [{side_short}] {status:>6}\n")
                
        self.timers_text.config(state=tk.DISABLED)
        
        # Auto-refresh every second
        self.root.after(1000, self.update_timers_display)

def main():
    print("=" * 60)
    print("Jungler Pathing Helper - MANUAL INPUT VERSION")
    print("=" * 60)
    print("\n⚠️  IMPORTANT - READ THIS:")
    print("This tool requires MANUAL input. You MUST press F1-F6")
    print("when you visually see the enemy jungler take a camp.")
    print("\nHOTKEYS:")
    for key, (name, _, side) in CAMPS.items():
        print(f"  {key}: {name} ({side})")
    print("\nThis tool is LEGAL because:")
    print("  ✅ You provide all inputs manually")
    print("  ✅ It only tracks information you witnessed")
    print("  ✅ No automation or memory reading")
    print("\nStarting overlay...")
    
    app = JunglerPathingHelper()
    app.root.mainloop()

if __name__ == "__main__":
    main()