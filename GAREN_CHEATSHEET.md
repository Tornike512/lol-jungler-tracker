# 🛡️ Garen Training Cheat Sheet

## Quick Commands

```bash
# Install missing packages
pip install gymnasium pynput

# Test setup (10 seconds, no League needed)
python train_garen.py --demo

# Train Stage 1: Farming
python train_garen.py --stage stage_1_farming

# Train Stage 2: Trading (after Stage 1)
python train_garen.py --stage stage_2_trading --resume checkpoints/garen/stage_1_farming/best_model.pt

# Train Stage 3: All-in (after Stage 2)
python train_garen.py --stage stage_3_all_in --resume checkpoints/garen/stage_2_trading/best_model.pt

# Train Stage 4: Macro (after Stage 3)
python train_garen.py --stage stage_4_macro --resume checkpoints/garen/stage_3_all_in/best_model.pt

# Monitor training
python dashboard.py

# Test trained model
python main.py infer --model checkpoints/garen/best_model.pt --practice-tool
```

## Training Stages

| Stage | Focus | Duration | Success Metric |
|-------|-------|----------|----------------|
| 1. Farming | Last-hitting minions | 2-3h | 50+ CS/10min |
| 2. Trading | Q-Auto-E combo | 3-4h | 60%+ win rate |
| 3. All-in | Full combo + R execute | 5-6h | 3+ kills, <2 deaths |
| 4. Macro | Split push, objectives | 8-10h | 2+ towers/game |

## League Setup

1. **Play** → **Training** → **Practice Tool**
2. Pick **Garen**
3. Press **Enter** to spawn minions
4. Stage 2+: Add enemy bot (Escape → Add bots)

## Common Issues

| Problem | Solution |
|---------|----------|
| Random actions | Normal for first hour - agent is exploring |
| "gymnasium not found" | `pip install gymnasium pynput` |
| Low FPS | Edit `src/config.py`: `TARGET_FPS = 30` |
| Agent clicks outside game | Check screen resolution in config |
| Training not improving | Check dashboard - reward should increase |

## Hotkeys

- **F12** - Emergency stop (kill switch)
- **Ctrl+C** - Save and exit training

## File Locations

- **Config**: `src/garen_config.py`
- **Checkpoints**: `checkpoints/garen/`
- **Logs**: `logs/`
- **Training script**: `train_garen.py`

## Monitoring Metrics

**Dashboard should show:**
- Episode Reward: Increasing ↗
- FPS: 50-60
- APM: 150-300 (human-like)
- HP/Mana: Valid percentages

## Garen Combos (What Agent Learns)

1. **Q-Auto**: Q for speed + silence, then auto-attack
2. **Q-Auto-E**: Basic trade combo
3. **Q-Auto-E-R**: All-in combo
4. **W-Retreat**: Block damage and heal with passive

## Expected Training Time

- **Quick test**: 1 hour per stage
- **Good performance**: 2-3 hours per stage
- **Competitive play**: 5+ hours per stage
- **Total**: ~20 hours for all stages

## Do I Need ROFL Files?

**No!** Agent learns by playing in real-time.

ROFL files are optional for advanced imitation learning (not implemented yet).

## Success Signs ✓

- Reward increasing over time
- Agent moves to low HP minions (Stage 1)
- Agent uses Q before auto-attacking (Stage 2)
- Agent saves R for low HP enemies (Stage 3)
- APM between 150-300
- Smooth mouse movements (Bezier curves)

## Failure Signs ✗

- Reward flat after 1+ hours
- APM too high (>500) or too low (<50)
- Agent stuck in one position
- FPS dropping below 30
- Loss not decreasing

## Quick Fixes

```bash
# Reduce lag
# Edit src/config.py:
TARGET_FPS = 30
YOLO_MODEL = "yolov8n.pt"

# Faster testing (fewer steps)
python train_garen.py --stage stage_1_farming --total-timesteps 50000

# Headless mode (no input, for testing)
python train_garen.py --stage stage_1_farming --headless
```

## Next Steps After Training

1. Test model: `python main.py infer --model checkpoints/garen/best_model.pt`
2. Train other champions (modify `garen_config.py`)
3. Add ROFL replay parsing for imitation learning
4. Fine-tune hyperparameters in `src/config.py`

---

**Remember**: First hour will look random - this is normal exploration!
