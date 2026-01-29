I need you to generate a complete Katarina (League of Legends) AI system for educational streaming purposes. This must be built with Python and divided into modular files. The AI will play Katarina at a high level while appearing human-like to avoid detection.

### PROJECT STRUCTURE:
Create the following files:

1. **config.py** - All constants, hotkeys, delays, champion-specific settings
2. **perception.py** - Screen capture using mss, YOLO object detection wrapper, minimap parsing, HUD reading (HP/abilities/gold)
3. **rofl_parser.py** - Parse .rofl replay files to extract Katarina training data (combos, positioning, dagger management)
4. **decision_engine.py** - Neural network (PyTorch) that takes game state and outputs actions. Must handle:
   - Dagger reset logic (E-W-Q-R combos)
   - All-in vs poke decisions
   - Wave management (when to roam)
   - Teamfight positioning (flanking)
5. **human_controller.py** - Execute actions with Bezier curve mouse movement, variable reaction times, skill-shot prediction with error margins
6. **katarina_mechanics.py** - Katarina-specific logic:
   - Dagger landing position calculation
   - Combo sequencing (E &gt; W &gt; Q &gt; AA &gt; R with proper cancels)
   - Reset detection (when to go in after kill)
   - Ward jumping for escape
7. **stream_overlay.py** - Tkinter overlay showing AI "thoughts" for OBS capture
8. **main.py** - Main loop integrating all components with safety checks
9. **train.py** - Script to train the model on ROFL data
10. **requirements.txt** - All dependencies

### SPECIFIC KATARINA MECHANICS TO IMPLEMENT:
- Dagger duration: 4 seconds on ground
- E range: 725 units
- W dagger lands 350 units behind cast position after 1.25s
- R can be cancelled by movement/E after 1.5s
- Q bounces to 3 additional targets
- Must track passive (Sinister Steel) procs

### TECHNICAL SPECS:
- Use YOLOv8 for vision (pre-trained, fine-tune later)
- Input resolution: 1920x1080
- Process at 10 FPS (human-like APM cap)
- Mouse movement: Bezier curves, never linear
- Reaction time: Log-normal distribution, mean 180ms
- Include "tilt" detection (if dying too much, play safer)

### DATA HANDLING:
- Load ROFL files from ./replays/ folder
- Extract state-action pairs every 0.1s during fights
- Feature engineering: distance_to_nearest_dagger, enemies_in_range, combo_damage_potential
- Save processed data to ./dataset/katarina_states.json

### SAFETY FEATURES:
- Check window title == "League of Legends (TM) Client"
- Detect if in Practice Tool (allow) vs Ranked (exit immediately)
- Hotkey to pause AI (F12)
- Visual indicator when AI is active (red border around screen)

### CODE STANDARDS:
- Type hints everywhere
- Docstrings for all functions
- Error handling with try/except blocks
- Logging to ./logs/ folder
- Async/await for I/O operations where appropriate

Generate all 10 files with complete, working code. Start with config.py, then perception.py, then the rest in dependency order. Include example usage in main.py that demonstrates a Practice Tool scenario.