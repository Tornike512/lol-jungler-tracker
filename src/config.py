"""
Configuration file for LoL RL Agent
Contains all hyperparameters, constants, and settings for the training and execution of the agent.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch


# ============================================================================
# SCREEN CAPTURE SETTINGS
# ============================================================================

@dataclass
class CaptureConfig:
    """Screen capture and ROI settings"""
    # Target resolution
    SCREEN_WIDTH: int = 1920
    SCREEN_HEIGHT: int = 1080
    TARGET_FPS: int = 60

    # Triple buffering for smooth capture
    BUFFER_COUNT: int = 3
    MAX_CAPTURE_LATENCY_MS: float = 5.0

    # Region of Interest (ROI) settings
    # Main viewport for combat/champion detection
    MAIN_VIEWPORT_WIDTH: int = 1280
    MAIN_VIEWPORT_HEIGHT: int = 720
    MAIN_VIEWPORT_X_OFFSET: int = 320  # Center of screen
    MAIN_VIEWPORT_Y_OFFSET: int = 180

    # Minimap region (bottom-right corner)
    MINIMAP_WIDTH_RATIO: float = 0.1425
    MINIMAP_HEIGHT_RATIO: float = 0.253
    MINIMAP_PADDING_RIGHT: int = 0
    MINIMAP_PADDING_BOTTOM: int = 0

    # HUD region (bottom center) for health/mana/abilities
    HUD_WIDTH: int = 500
    HUD_HEIGHT: int = 200
    HUD_Y_OFFSET: int = 880  # From top
    HUD_X_OFFSET: int = 710  # From left (centered)


# ============================================================================
# COMPUTER VISION SETTINGS
# ============================================================================

@dataclass
class VisionConfig:
    """YOLO and OCR settings"""
    # YOLO model settings
    YOLO_MODEL: str = "yolov8n.pt"  # nano model for speed
    YOLO_CONFIDENCE: float = 0.4
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_INFERENCE_TIME_MS: float = 16.0  # Must complete within one frame at 60fps

    # Detection classes
    CLASSES: Dict[str, int] = None  # Will be populated from dataset

    # Default classes for LoL
    DEFAULT_CLASSES: Tuple[str, ...] = (
        "enemy_champion",
        "ally_champion",
        "minion_melee",
        "minion_ranged",
        "minion_cannon",
        "minion_super",
        "turret_enemy",
        "turret_ally",
        "jungle_monster",
        "skillshot_projectile",
        "ward",
        "player_champion",
    )

    # OCR settings for ability cooldowns
    OCR_LANGUAGE: str = "eng"
    OCR_PSM_MODE: int = 6  # Assume uniform block of text

    # State vector dimension
    STATE_DIM: int = 512

    # Feature extraction settings
    MAX_ENTITIES_TRACKED: int = 30  # Maximum number of entities in state vector


# ============================================================================
# REINFORCEMENT LEARNING SETTINGS
# ============================================================================

@dataclass
class RLConfig:
    """Reinforcement learning hyperparameters"""
    # Algorithm selection
    ALGORITHM: str = "PPO"  # Options: "PPO", "SAC"

    # PPO-specific hyperparameters
    PPO_LEARNING_RATE: float = 3e-4
    PPO_N_STEPS: int = 2048
    PPO_BATCH_SIZE: int = 64
    PPO_N_EPOCHS: int = 10
    PPO_GAMMA: float = 0.99
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_RANGE: float = 0.2
    PPO_ENT_COEF: float = 0.01
    PPO_VF_COEF: float = 0.5
    PPO_MAX_GRAD_NORM: float = 0.5

    # SAC-specific hyperparameters
    SAC_LEARNING_RATE: float = 3e-4
    SAC_BUFFER_SIZE: int = 1_000_000
    SAC_LEARNING_STARTS: int = 10_000
    SAC_BATCH_SIZE: int = 256
    SAC_TAU: float = 0.005
    SAC_GAMMA: float = 0.99
    SAC_TRAIN_FREQ: int = 1
    SAC_GRADIENT_STEPS: int = 1

    # Network architecture
    POLICY_NETWORK_ARCH: Tuple[int, ...] = (512, 512, 256, 256)
    VALUE_NETWORK_ARCH: Tuple[int, ...] = (512, 512, 256)

    # LSTM for temporal memory
    USE_LSTM: bool = True
    LSTM_HIDDEN_SIZE: int = 256
    LSTM_NUM_LAYERS: int = 2
    SEQUENCE_LENGTH: int = 16  # Number of frames to remember

    # Training settings
    TOTAL_TIMESTEPS: int = 10_000_000
    CHECKPOINT_FREQ: int = 50_000
    LOG_INTERVAL: int = 10
    EVAL_FREQ: int = 10_000
    EVAL_EPISODES: int = 5

    # Curriculum learning stages
    CURRICULUM_STAGES: Tuple[str, ...] = ("cs_training", "trading", "objectives")
    STAGE_THRESHOLDS: Dict[str, float] = None

    def __post_init__(self):
        if self.STAGE_THRESHOLDS is None:
            self.STAGE_THRESHOLDS = {
                "cs_training": 50.0,  # Average CS per 10 min
                "trading": 5.0,  # Average kills - deaths
                "objectives": 3.0,  # Average objectives taken per game
            }


# ============================================================================
# ACTION SPACE CONFIGURATION
# ============================================================================

@dataclass
class ActionConfig:
    """Action space and input execution settings"""
    # Action space dimensions
    # Continuous actions
    MOUSE_X_MIN: float = -1.0
    MOUSE_X_MAX: float = 1.0
    MOUSE_Y_MIN: float = -1.0
    MOUSE_Y_MAX: float = 1.0
    CAMERA_PAN_X_MIN: float = -1.0
    CAMERA_PAN_X_MAX: float = 1.0
    CAMERA_PAN_Y_MIN: float = -1.0
    CAMERA_PAN_Y_MAX: float = 1.0

    # Discrete actions
    MOUSE_BUTTONS: Tuple[str, ...] = ("none", "left", "right")
    KEYBOARD_KEYS: Tuple[str, ...] = (
        "none", "q", "w", "e", "r", "d", "f",
        "ctrl+q", "ctrl+w", "ctrl+e", "ctrl+r",  # Level up abilities
        "1", "2", "3", "4", "5", "6", "7",  # Item hotkeys
        "b", "tab", "spacebar",  # Recall, scoreboard, camera lock
    )

    # Total action space size
    ACTION_DIM_CONTINUOUS: int = 4  # mouse_x, mouse_y, camera_x, camera_y
    ACTION_DIM_DISCRETE_MOUSE: int = 3  # none, left, right
    ACTION_DIM_DISCRETE_KEYBOARD: int = 21  # number of keyboard options

    # Input timing and humanization
    # NOTE: For faster training, set MIN_REACTION_TIME_MS to 20-50ms
    # For human-like play (inference), use 150ms+
    MIN_REACTION_TIME_MS: float = 30.0   # Reduced for training speed
    AVG_REACTION_TIME_MS: float = 50.0   # Reduced for training speed
    STD_REACTION_TIME_MS: float = 15.0   # Reduced for training speed

    # Mouse movement settings
    USE_BEZIER_CURVES: bool = True
    BEZIER_CONTROL_POINTS: int = 4
    MOUSE_SPEED_MIN: float = 0.5  # pixels per ms
    MOUSE_SPEED_MAX: float = 3.0

    # APM (Actions Per Minute) limits
    MIN_APM: int = 80
    MAX_APM: int = 500
    TARGET_APM: int = 250

    # Debounce timing (prevent double-clicks)
    CLICK_DEBOUNCE_MS: float = 80.0
    KEY_DEBOUNCE_MS: float = 100.0


# ============================================================================
# REWARD SHAPING
# ============================================================================

@dataclass
class RewardConfig:
    """Reward function coefficients for curriculum learning"""
    # Stage 1: CS Training
    REWARD_CS_HIT: float = 1.0
    REWARD_CS_MISS_CANNON: float = -0.1
    REWARD_LEVEL_UP: float = 0.5
    PENALTY_IDLE_PER_SECOND: float = -0.01

    # ==========================================================================
    # SHAPING REWARDS (guide agent toward farming behavior)
    # These are smaller rewards that help the agent discover the main CS reward
    # ==========================================================================

    # Lane presence - reward for being in the right area of the screen
    # In LoL, the lane is roughly in the center-bottom area during laning phase
    REWARD_LANE_PRESENCE: float = 0.005  # Small continuous reward for being in lane

    # Proximity to minions - reward for being close to enemy minions
    REWARD_NEAR_MINIONS: float = 0.01  # Per minion within attack range
    MINION_PROXIMITY_THRESHOLD: float = 0.3  # Normalized distance (0-1 scale)

    # Targeting reward - reward for cursor being near minions
    REWARD_CURSOR_NEAR_MINION: float = 0.005  # Mouse near a minion
    CURSOR_TARGET_THRESHOLD: float = 0.1  # Normalized distance for "targeting"

    # Attack action reward - small reward for right-clicking (attack command)
    REWARD_ATTACK_ACTION: float = 0.002  # Encourage right-click actions

    # Movement toward minions - reward for reducing distance to nearest minion
    REWARD_APPROACH_MINION: float = 0.008  # Moving closer to minions

    # Low HP minion targeting - bonus for cursor near low HP minion
    REWARD_TARGET_LOW_HP: float = 0.02  # Targeting minion that might be last-hittable

    # Screen center preference - encourage staying in playable area
    REWARD_SCREEN_CENTER: float = 0.002  # Being in center of screen (where action is)

    # Stage 2: Trading
    REWARD_DAMAGE_DEALT: float = 0.02  # Per point of damage
    PENALTY_DAMAGE_TAKEN: float = -0.03
    REWARD_KILL: float = 5.0
    PENALTY_DEATH: float = -10.0
    REWARD_ASSIST: float = 2.0

    # Stage 3: Objectives
    REWARD_TOWER_PLATE: float = 10.0
    REWARD_TOWER_DESTROY: float = 25.0
    REWARD_DRAGON: float = 20.0
    REWARD_BARON: float = 40.0
    REWARD_HERALD: float = 15.0
    PENALTY_OBJECTIVE_LOST: float = -15.0

    # Global penalties
    PENALTY_AFK_DETECTED: float = -100.0
    PENALTY_DISCONNECT: float = -100.0

    # Exploration bonuses
    BONUS_NEW_AREA_EXPLORED: float = 0.1
    BONUS_VISION_SCORE: float = 0.05  # Per vision score point


# ============================================================================
# SAFETY AND ANTI-DETECTION
# ============================================================================

@dataclass
class SafetyConfig:
    """Safety features and anti-detection measures"""
    # Kill switch
    KILL_SWITCH_KEY: str = "F12"
    EMERGENCY_STOP_KEY: str = "ESC"

    # Game state validation
    REQUIRED_WINDOW_TITLE: str = "League of Legends (TM) Client"
    ALLOWED_GAME_MODES: Tuple[str, ...] = ("PRACTICETOOL", "CUSTOM")
    FORBIDDEN_GAME_MODES: Tuple[str, ...] = ("RANKED", "NORMAL", "ARAM")

    # Input validation
    VALIDATE_SCREEN_STATE: bool = True
    VERIFY_ACTION_EXECUTION: bool = True

    # Timing randomization to appear human
    USE_RANDOM_DELAYS: bool = True
    USE_RANDOM_MOUSE_PATHS: bool = True
    USE_RANDOM_APM_VARIATION: bool = True

    # Never use sub-human reaction times
    # Set to False during training for faster iterations
    ENFORCE_MIN_REACTION_TIME: bool = False  # Disabled for training speed

    # Session management
    MAX_GAME_DURATION_MINUTES: int = 60
    AUTO_STOP_ON_GAME_END: bool = True
    REQUIRE_MANUAL_QUEUE: bool = True  # Never auto-queue into games


# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Logging and visualization settings"""
    # Directories
    LOG_DIR: str = "logs"
    CHECKPOINT_DIR: str = "checkpoints"
    SCREENSHOT_DIR: str = "screenshots"
    TENSORBOARD_DIR: str = "runs"

    # Logging levels
    CONSOLE_LOG_LEVEL: str = "INFO"
    FILE_LOG_LEVEL: str = "DEBUG"

    # Dashboard settings
    SHOW_DASHBOARD: bool = True
    DASHBOARD_UPDATE_FREQ_MS: int = 100
    SHOW_DETECTIONS_OVERLAY: bool = True
    SHOW_QVALUES: bool = True
    SHOW_REWARD_PLOT: bool = True

    # Screenshot settings
    SAVE_SCREENSHOTS: bool = True
    SCREENSHOT_INTERVAL_STEPS: int = 1000

    # Performance metrics
    TRACK_FPS: bool = True
    TRACK_LATENCY: bool = True
    TRACK_APM: bool = True


# ============================================================================
# GAME-SPECIFIC CONSTANTS
# ============================================================================

@dataclass
class GameConstants:
    """League of Legends game constants"""
    # Map dimensions (Summoner's Rift)
    MAP_WIDTH: int = 14820
    MAP_HEIGHT: int = 14881

    # Turret attack range
    TURRET_RANGE: int = 775

    # Experience and gold
    MINION_GOLD_MELEE: int = 21
    MINION_GOLD_RANGED: int = 14
    MINION_GOLD_CANNON: int = 60

    # Respawn timers (seconds)
    CHAMPION_BASE_RESPAWN: float = 10.0
    CHAMPION_RESPAWN_PER_LEVEL: float = 2.5

    # Objectives
    DRAGON_RESPAWN_TIME: float = 300.0
    BARON_RESPAWN_TIME: float = 360.0
    HERALD_RESPAWN_TIME: float = 360.0


# ============================================================================
# GLOBAL CONFIG INSTANCES
# ============================================================================

capture_cfg = CaptureConfig()
vision_cfg = VisionConfig()
rl_cfg = RLConfig()
action_cfg = ActionConfig()
reward_cfg = RewardConfig()
safety_cfg = SafetyConfig()
logging_cfg = LoggingConfig()
game_cfg = GameConstants()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device() -> torch.device:
    """Get the appropriate torch device (CUDA if available)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config_dict() -> Dict[str, Any]:
    """Get all config as a dictionary for logging"""
    return {
        "capture": capture_cfg.__dict__,
        "vision": vision_cfg.__dict__,
        "rl": rl_cfg.__dict__,
        "action": action_cfg.__dict__,
        "reward": reward_cfg.__dict__,
        "safety": safety_cfg.__dict__,
        "logging": logging_cfg.__dict__,
        "game": game_cfg.__dict__,
    }


if __name__ == "__main__":
    # Print configuration summary
    import json
    print("=" * 80)
    print("LoL RL Agent Configuration")
    print("=" * 80)
    print(json.dumps(get_config_dict(), indent=2, default=str))
