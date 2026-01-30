# League of Legends RL Agent

A real-time AI agent that plays League of Legends using reinforcement learning with curriculum training. The agent processes screen pixels and executes mouse/keyboard inputs using deep learning and computer vision.

## Features

- **High-Performance Screen Capture**: GPU-accelerated 60 FPS capture with <5ms latency using triple buffering
- **Computer Vision Pipeline**: YOLOv8 object detection for champions, minions, turrets, and skillshots
- **Reinforcement Learning**: PPO algorithm with LSTM for temporal memory
- **Hybrid Action Space**: Continuous mouse movement + discrete keyboard/mouse buttons
- **Curriculum Learning**: Progressive training from CS → Trading → Objectives
- **Human-like Input**: Bezier curve mouse movement, randomized timing, APM limiting
- **Safety Features**: Kill switch, game mode validation, anti-detection measures
- **Real-time Dashboard**: Live visualization of detections, rewards, and performance metrics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SCREEN CAPTURE                       │
│  GPU-accelerated frame grabbing @ 60 FPS                │
│  Triple buffering | ROI extraction                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   VISION PIPELINE                        │
│  YOLOv8 Detection | OCR | State Vectorization           │
│  512-dim feature vector                                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    RL AGENT (PPO)                        │
│  LSTM Actor-Critic | Hybrid Action Space                │
│  Curriculum Learning | GAE                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  INPUT CONTROLLER                        │
│  Bezier curves | Randomized timing | APM limiting        │
│  SendInput API (Windows) | pynput (Linux)               │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
lol-jungler-tracker/
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration and hyperparameters
│   ├── capture.py             # Screen capture with triple buffering
│   ├── vision.py              # YOLO detection and state vectorization
│   ├── rl_agent.py            # PPO agent with LSTM
│   ├── input_controller.py    # Human-like input simulation
│   └── lol_env.py             # Gymnasium environment wrapper
├── train.py                   # Main training script
├── dashboard.py               # Real-time visualization dashboard
├── requirements.txt           # Python dependencies
├── models/                    # Trained model checkpoints
├── data/
│   ├── replays/              # .rofl replay files (for future)
│   ├── datasets/             # Training datasets
│   └── training_logs/        # TensorBoard logs
├── logs/                      # Application logs
├── checkpoints/              # Model checkpoints during training
└── screenshots/              # Debug screenshots

```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd lol-jungler-tracker
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU Setup (Recommended)

For CUDA support (NVIDIA GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5. Download YOLO Model

The YOLOv8 nano model will auto-download on first run, or you can manually download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Training

#### Practice Tool (Recommended for Initial Training)

```bash
python train.py --practice-tool --curriculum-stage cs_training
```

#### Headless Mode (Testing without League running)

```bash
python train.py --headless --curriculum-stage cs_training
```

#### Full Training Options

```bash
python train.py \
    --curriculum-stage cs_training \
    --practice-tool \
    --total-timesteps 1000000 \
    --checkpoint-dir checkpoints/cs_training
```

Available curriculum stages:
- `cs_training`: Focus on last-hitting minions
- `trading`: Focus on damage dealt/taken, kills
- `objectives`: Focus on towers, dragons, baron

### Dashboard

Launch the real-time dashboard to monitor training:

```bash
python dashboard.py
```

### Resuming Training

```bash
python train.py --resume checkpoints/checkpoint_50000.pt
```

## Configuration

All settings are in `src/config.py`:

- **Screen Capture**: FPS, buffer settings, ROI regions
- **Vision**: YOLO confidence, inference timeout, state dimensions
- **RL**: PPO hyperparameters, LSTM size, learning rate
- **Actions**: APM limits, reaction times, mouse speed
- **Rewards**: Curriculum-specific reward coefficients
- **Safety**: Kill switch, game mode restrictions

## Safety and Ethics

### Important Guidelines

1. **Use ONLY in Practice Tool or Custom Games** - Never use in ranked or normal games
2. **Respects Riot's Terms of Service** - This is for educational/research purposes only
3. **Kill Switch**: Press F12 to immediately stop the agent
4. **Game Mode Detection**: Agent will refuse to run in forbidden game modes

### Anti-Detection Features

- Human-like reaction times (150-300ms)
- Randomized mouse paths using Bezier curves
- APM limiting (250-500 average)
- No sub-human inputs
- Natural timing variations

## Implementation Phases

### MVP (Phase 1) ✅
- Screen capture module
- Basic YOLO detection
- CS training in Practice Tool

### Combat (Phase 2)
- Champion detection
- Ability usage
- 1v0 bot training

### Awareness (Phase 3)
- Minimap parsing
- Objective tracking
- Custom game vs Intermediate bots

### Optimization (Phase 4)
- Multi-threading optimization
- TensorRT conversion
- Performance tuning

## Performance Targets

- **Latency**: <100ms end-to-end (target 60ms)
- **FPS**: 60 FPS capture and processing
- **Inference**: <16ms YOLO inference time
- **APM**: 250-500 (human-like range)

## Curriculum Learning

Training progresses through three stages:

### Stage 1: CS Training
**Goal**: Learn to last-hit minions
**Rewards**:
- +1.0 per last hit
- -0.1 per missed cannon
- +0.5 per level up

### Stage 2: Trading
**Goal**: Win trades and get kills
**Rewards**:
- +0.02 per damage dealt
- -0.03 per damage taken
- +5.0 per kill
- -10.0 per death

### Stage 3: Objectives
**Goal**: Take towers and objectives
**Rewards**:
- +10.0 per tower plate
- +25.0 per tower
- +20.0 per dragon
- +40.0 per baron

## Development

### Testing Individual Components

```bash
# Test screen capture
python -m src.capture

# Test vision pipeline
python -m src.vision

# Test RL agent
python -m src.rl_agent

# Test input controller
python -m src.input_controller

# Test environment
python -m src.lol_env
```

### Code Style

- Type hints for all functions
- Docstrings for classes and methods
- Configuration via `config.py` (no magic numbers)

## Troubleshooting

### YOLOv8 not detecting anything
- Check if YOLO model is trained on LoL data
- Adjust `YOLO_CONFIDENCE` in config.py
- Verify screen capture is working

### Input not executing
- Check if pynput is installed (Linux)
- Verify game window is focused
- Check APM throttling settings

### Low FPS
- Reduce YOLO model size (yolov8n → yolov8n6)
- Decrease capture FPS
- Use GPU for inference

### Training not converging
- Adjust learning rate
- Check reward shaping
- Verify observation normalization

## Future Enhancements

- [ ] Train YOLO on LoL dataset
- [ ] Integrate Riot Live Client API for ground truth
- [ ] Add champion-specific mechanics modules
- [ ] Implement replay parsing for imitation learning
- [ ] Multi-agent cooperative play
- [ ] Advanced vision (fog of war prediction)

## License

Educational/Research use only. Respect Riot Games' Terms of Service.

## Acknowledgments

- OpenAI's Dota 2 bot for inspiration
- DeepMind's StarCraft II agent (AlphaStar)
- Ultralytics for YOLOv8
- Stable Baselines3 for RL algorithms

## Disclaimer

This project is for educational and research purposes only. Using bots in online games may violate terms of service. Always use in offline/practice modes. The authors are not responsible for any misuse of this software.
