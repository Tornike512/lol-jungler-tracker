# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

### 2. Test Installation

```bash
# Run a quick demo (no League required, headless mode)
python main.py demo
```

This will test:
- Screen capture ✓
- Vision pipeline ✓
- RL agent ✓
- Performance metrics ✓

### 3. Train Your First Agent

#### Option A: Practice Tool (Recommended)

1. Open League of Legends
2. Start a Practice Tool game
3. Run training:

```bash
python train.py --practice-tool --curriculum-stage cs_training
```

#### Option B: Test Without League (Headless)

```bash
# This runs the training loop without actually playing
python train.py --headless --curriculum-stage cs_training --total-timesteps 10000
```

### 4. Monitor Training

In another terminal:

```bash
python dashboard.py
```

This shows:
- Current game state (HP, mana, detections)
- Performance metrics (FPS, latency, APM)
- Training progress (rewards, episode length)

## Common Commands

### Run Demo
```bash
python main.py demo
```

### Train Agent
```bash
# CS training (last-hitting)
python train.py --practice-tool --curriculum-stage cs_training

# Trading (combat)
python train.py --practice-tool --curriculum-stage trading

# Objectives (towers, dragons)
python train.py --practice-tool --curriculum-stage objectives
```

### Run Trained Model
```bash
python main.py infer --model checkpoints/best_model.pt --practice-tool
```

### Resume Training
```bash
python train.py --resume checkpoints/checkpoint_50000.pt
```

## Curriculum Learning Path

1. **CS Training** (2-5 hours)
   - Goal: 50+ CS per 10 minutes
   - Focus: Last-hitting mechanics
   - Checkpoint: `checkpoints/cs_training/`

2. **Trading** (5-10 hours)
   - Goal: Positive KDA
   - Focus: Combat decisions
   - Checkpoint: `checkpoints/trading/`

3. **Objectives** (10+ hours)
   - Goal: Take towers and dragons
   - Focus: Macro decisions
   - Checkpoint: `checkpoints/objectives/`

## Troubleshooting

### "ultralytics not found"
```bash
pip install ultralytics
```

### "pynput not found" (Linux)
```bash
pip install pynput
```

### Low FPS
Edit `src/config.py`:
```python
TARGET_FPS = 30  # Reduce from 60
YOLO_MODEL = "yolov8n.pt"  # Use nano model
```

### Agent not acting
- Check if headless mode is disabled: `--no-headless` flag
- Verify pynput is installed
- Check game window is focused

## Safety Reminders

- **Always use in Practice Tool or Custom games**
- **Never use in ranked or normal games**
- **Press F12 to emergency stop**
- **Respect Riot's Terms of Service**

## Next Steps

1. Read the full [README.md](README.md) for detailed information
2. Check [src/config.py](src/config.py) for all settings
3. Explore the code in `src/` directory
4. Join discussions and share results!

## File Structure Quick Reference

```
train.py              - Main training script
main.py               - Run trained models
dashboard.py          - Real-time visualization
src/
  ├── config.py       - All settings (start here!)
  ├── capture.py      - Screen capture
  ├── vision.py       - YOLO detection
  ├── rl_agent.py     - PPO algorithm
  ├── input_controller.py - Input execution
  └── lol_env.py      - Environment wrapper
```

## Getting Help

1. Check [README.md](README.md) for detailed documentation
2. Review error messages carefully
3. Test individual components: `python -m src.capture`
4. Check logs in `logs/` directory

Happy training! 🎮🤖
