# 🛡️ Garen Training Guide

Complete step-by-step guide to train an RL agent that plays Garen.

## 📋 Quick Overview

**No ROFL files needed to start!** The agent learns by playing in real-time.

The training uses **curriculum learning** with 4 stages:
1. **Stage 1**: Farming (learn to CS)
2. **Stage 2**: Trading (Q-Auto-E combo)
3. **Stage 3**: All-in (full combo with R)
4. **Stage 4**: Macro (split push, objectives)

Each stage takes 1-4 hours depending on your hardware.

---

## 🚀 Step-by-Step Training Process

### Prerequisites

1. Install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

2. Fix missing packages (from diagnostics):
```bash
pip install gymnasium pynput
```

### Stage 1: Farming (CSing)

**Goal**: Teach Garen to last-hit minions

#### Step 1: Open League of Legends

1. Start League client
2. Click "Play" → "Training" → "Practice Tool"
3. Select **Garen** as your champion
4. Start the game
5. Once loaded in-game, press Enter to spawn minions

#### Step 2: Test Your Setup

Run a 10-second demo first to make sure everything works:

```bash
python train_garen.py --demo
```

This will:
- ✓ Test screen capture
- ✓ Test vision pipeline
- ✓ Test RL agent
- ✓ Show performance metrics

If you see "Demo completed successfully!", you're ready!

#### Step 3: Start Training

```bash
python train_garen.py --stage stage_1_farming
```

**What happens:**
- Agent starts watching your screen
- Tries random mouse movements and key presses
- Gets +1.5 reward for each minion last-hit
- Gets +3.0 reward for cannon minions
- Gradually learns to move to low HP minions and attack

**Expected behavior:**
- First 10 minutes: Random actions, maybe 0-10 CS
- After 30 minutes: Starting to move towards minions, 10-20 CS
- After 1 hour: Somewhat consistent last-hitting, 30-40 CS
- After 2-3 hours: Good CSing, 50+ CS per 10 min ✓

#### Step 4: Monitor Progress

In another terminal, launch dashboard:

```bash
python dashboard.py
```

Watch:
- **Episode Reward**: Should gradually increase
- **APM**: Should be 150-300 (human-like)
- **FPS**: Should be 50-60

**Training tips:**
- Keep the game window visible (don't minimize)
- Don't move your mouse (agent controls it)
- You can pause/resume League to reset minion waves
- Press **F12** for emergency stop
- Press **Ctrl+C** in terminal to save and quit

#### Step 5: Verify Stage 1 Success

After 100,000 steps (~2-3 hours), check:
- Average reward: Should be positive (>0)
- CS rate: Should hit 50+ per 10 minutes
- Checkpoint saved: `checkpoints/garen/stage_1_farming/best_model.pt`

---

### Stage 2: Trading (Q-Auto-E Combo)

**Goal**: Learn to trade using Q → Auto → E combo

#### Step 1: Continue from Stage 1

```bash
# Resume from previous checkpoint
python train_garen.py --stage stage_2_trading --resume checkpoints/garen/stage_1_farming/best_model.pt
```

OR start fresh:

```bash
python train_garen.py --stage stage_2_trading
```

#### Step 2: Add Enemy Champion

In League Practice Tool:
1. Press Escape → "Add bots"
2. Add 1 enemy champion (any champion)
3. Place them in lane

**What happens:**
- Agent learns to approach enemy
- Tries pressing Q (speed boost + damage)
- Auto-attacks after Q
- Activates E (spin) for extended trades
- Gets reward for successful Q-Auto combos

**Training duration**: 200,000 steps (~3-4 hours)

**Success metrics:**
- Win 60%+ of trades (more damage dealt than taken)
- Successfully execute Q-Auto-E combo
- Use W to block burst damage

---

### Stage 3: All-in with Ultimate

**Goal**: Learn full combo and when to use R (execute)

```bash
python train_garen.py --stage stage_3_all_in --resume checkpoints/garen/stage_2_trading/best_model.pt
```

**What happens:**
- Learns to recognize when enemy is low HP (<25%)
- Uses full combo: Q → Auto → E → R
- Gets +10 reward for R executes
- Gets -5 penalty for wasting R on full HP enemies

**Training duration**: 300,000 steps (~5-6 hours)

**Success metrics:**
- 3+ kills per game
- <2 deaths per game
- Efficient R usage (mostly on low HP targets)

---

### Stage 4: Macro & Objectives

**Goal**: Split push, take towers, control objectives

```bash
python train_garen.py --stage stage_4_macro --resume checkpoints/garen/stage_3_all_in/best_model.pt
```

**What happens:**
- Learns to push lanes
- Takes tower plates and towers
- Positions for objectives (Herald, Dragon)
- Makes macro decisions

**Training duration**: 500,000 steps (~8-10 hours)

**Success metrics:**
- Take 2+ towers per game
- Efficient wave clearing with E
- Good map positioning

---

## 📊 Complete Training Timeline

| Stage | Duration | Total Time | Checkpoint |
|-------|----------|------------|------------|
| 1. Farming | 2-3 hours | 2-3 hours | `checkpoints/garen/stage_1/` |
| 2. Trading | 3-4 hours | 5-7 hours | `checkpoints/garen/stage_2/` |
| 3. All-in | 5-6 hours | 10-13 hours | `checkpoints/garen/stage_3/` |
| 4. Macro | 8-10 hours | 18-23 hours | `checkpoints/garen/stage_4/` |

**Total**: ~20 hours of training for a competent Garen bot

---

## 🎮 Using Your Trained Garen Bot

After training, test it:

```bash
python main.py infer --model checkpoints/garen/best_model.pt --practice-tool
```

This will:
1. Load your trained model
2. Watch the game
3. Control Garen using learned policy
4. Run for one full game

---

## 🔧 Troubleshooting

### "Agent is doing random actions"
- **Normal** for first 30-60 minutes of training
- Agent is exploring to learn what works
- Check dashboard: reward should gradually increase

### "No detections / State vector all zeros"
- YOLO might not be detecting anything yet
- Default YOLOv8 uses COCO dataset (not LoL-specific)
- Agent can still learn from HUD data (HP, position)

### "Training is very slow"
- Reduce FPS: Edit `src/config.py`, set `TARGET_FPS = 30`
- Use smaller YOLO: Set `YOLO_MODEL = "yolov8n.pt"`
- Reduce timesteps: `--total-timesteps 50000` for testing

### "Agent clicks outside game window"
- Check `src/config.py` screen resolution matches yours
- Make sure game is in windowed mode (not fullscreen)

### "ImportError: gymnasium"
```bash
pip install gymnasium pynput
```

---

## 📈 Monitoring Training Quality

### Good signs:
✓ Episode reward increasing over time
✓ APM between 150-300
✓ FPS stable at 50-60
✓ Agent focuses on low HP minions (Stage 1)
✓ Agent approaches enemies before using abilities (Stage 2+)

### Bad signs:
✗ Episode reward staying flat/negative after 1 hour
✗ APM too high (>500) - might be spamming
✗ APM too low (<50) - might be stuck
✗ Loss not decreasing - check learning rate

---

## 🎯 Advanced: Adding ROFL Replay Learning (Optional)

If you want to learn from expert Garen replays:

### Step 1: Collect Replays

1. Download Garen replays from:
   - Your own games (`Documents/League of Legends/Highlights/`)
   - [op.gg](https://www.op.gg) - search high elo Garen players
   - [replay.lol](https://replay.lol)

2. Save `.rofl` files to `data/replays/garen/`

### Step 2: Parse Replays (TODO - Not Yet Implemented)

This requires additional work to:
- Parse .rofl files
- Extract game states at each timestamp
- Record actions taken (mouse position, keys pressed)
- Create dataset for imitation learning

I can help you implement this if you want to pursue this approach!

### Step 3: Pre-train on Replays

Use imitation learning to teach agent to mimic expert play, then fine-tune with RL.

**Pros:**
- Faster convergence
- Learn from expert strategies
- Better initial policy

**Cons:**
- Requires significant setup
- Need quality replays
- More complex pipeline

---

## 🎓 Training Tips

### 1. Start Small
- Begin with stage_1_farming only
- Don't skip stages
- Verify each stage works before progressing

### 2. Use Checkpoints
- Always resume from previous stage
- Save frequently (automatic every 50k steps)
- Keep best_model.pt separate

### 3. Monitor Dashboard
- Check if reward is increasing
- Watch APM to ensure human-like behavior
- Verify FPS is stable

### 4. Be Patient
- First hour: mostly random exploration
- After 2-3 hours: should see clear improvement
- Full training: 20+ hours for strong performance

### 5. Practice Tool Setup
- Spawn minions: Press Enter
- Reset game state: Escape → "Reset game"
- Add bots for stage 2+
- Remove fog of war: Makes vision easier

---

## 📝 Summary

**To train Garen:**

```bash
# 1. Install dependencies
pip install gymnasium pynput

# 2. Start League → Practice Tool → Pick Garen

# 3. Test setup
python train_garen.py --demo

# 4. Start training Stage 1
python train_garen.py --stage stage_1_farming

# 5. Monitor in another terminal
python dashboard.py

# 6. After 2-3 hours, progress to Stage 2
python train_garen.py --stage stage_2_trading --resume checkpoints/garen/stage_1_farming/best_model.pt

# 7. Continue through all 4 stages

# 8. Test your trained agent
python main.py infer --model checkpoints/garen/best_model.pt --practice-tool
```

**No ROFL files needed!** Agent learns by playing in real-time.

---

## 🤔 Still Have Questions?

- Check `src/garen_config.py` for Garen-specific settings
- Read `README.md` for architecture details
- Review `src/config.py` for hyperparameters
- Test individual components: `python -m src.capture`

Good luck training! 🛡️⚔️
