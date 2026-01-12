# LoL Jungler Tracker

A Python application that tracks the enemy jungler on the League of Legends minimap and provides voice alerts. Designed for mid lane players to improve map awareness.

## Features

- **Auto-detect enemy jungler** via Riot Live Client API (finds player with Smite)
- **Minimap tracking** with resolution-independent detection
- **Voice alerts** when jungler disappears or appears in dangerous zones
- **Prediction system** estimates jungler location when not visible
- **On-screen overlay** showing tracking status
- **Logging** all detections to CSV/JSON

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- Windows (for overlay and game detection)
- League of Legends

## Usage

```bash
python main.py
```

1. Start the tracker
2. Launch a League of Legends game
3. Tracker auto-detects the game and enemy jungler
4. Voice alerts play when jungler disappears from vision

## How It Works

- Uses Riot's Live Client API (`https://127.0.0.1:2999/liveclientdata/allgamedata`)
- Scans minimap 15 times per second
- Template matching to detect champion icons
- 10-second cooldown between alerts

## Project Structure

```
lol-jungler-tracker/
├── main.py          # Entry point, CLI
├── capture.py       # Screen capture, minimap detection
├── detector.py      # Champion detection, Riot API
├── predictor.py     # Jungle pathing prediction
├── zones.py         # Map zone definitions
├── voice.py         # TTS alerts
├── overlay.py       # On-screen overlay
├── logger.py        # Detection logging
├── config.py        # Settings
└── requirements.txt
```