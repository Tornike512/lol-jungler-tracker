"""
League of Legends Gymnasium Environment
Wraps the capture, vision, and input systems into a standard RL environment.
"""
import time
import threading
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

from .config import vision_cfg, action_cfg, reward_cfg, safety_cfg, capture_cfg
from .capture import ScreenCapture, FrameData
from .vision import VisionPipeline, GameState
from .input_controller import InputController
from .cs_detector import CSDetector

# Try to import Live Client API (preferred over OCR)
try:
    from .live_client_api import LiveClientCSDetector, LiveClientAPI
    LIVE_CLIENT_AVAILABLE = True
except ImportError:
    LIVE_CLIENT_AVAILABLE = False

# Global kill switch flag
_kill_switch_pressed = False
_kill_switch_listener = None


def _start_kill_switch_listener():
    """Start a background listener for the kill switch (F12)"""
    global _kill_switch_listener, _kill_switch_pressed

    try:
        from pynput import keyboard

        def on_press(key):
            global _kill_switch_pressed
            try:
                if key == keyboard.Key.f12:
                    print("\n\n*** F12 KILL SWITCH ACTIVATED ***")
                    print("Stopping training gracefully...")
                    _kill_switch_pressed = True
                    return False  # Stop listener
            except:
                pass

        _kill_switch_listener = keyboard.Listener(on_press=on_press)
        _kill_switch_listener.start()
        print(f"Kill switch enabled: Press {safety_cfg.KILL_SWITCH_KEY} to stop")

    except Exception as e:
        print(f"Warning: Could not start kill switch listener: {e}")


def is_kill_switch_pressed() -> bool:
    """Check if kill switch was pressed"""
    return _kill_switch_pressed


def reset_kill_switch():
    """Reset kill switch state"""
    global _kill_switch_pressed
    _kill_switch_pressed = False


class LoLEnvironment(gym.Env):
    """
    Gymnasium environment for League of Legends RL training.

    Observation space: 512-dim feature vector from vision pipeline
    Action space: Hybrid continuous-discrete (mouse movement + clicks + keyboard)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        curriculum_stage: str = "cs_training",
        use_yolo: bool = True,
        headless: bool = False
    ):
        super().__init__()

        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        self.headless = headless

        # Define observation space (state vector from vision)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vision_cfg.STATE_DIM,),
            dtype=np.float32
        )

        # Define action space (hybrid)
        # Continuous: [mouse_x, mouse_y, camera_x, camera_y]
        self.action_space = spaces.Dict({
            "continuous": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_cfg.ACTION_DIM_CONTINUOUS,),
                dtype=np.float32
            ),
            "discrete_mouse": spaces.Discrete(action_cfg.ACTION_DIM_DISCRETE_MOUSE),
            "discrete_keyboard": spaces.Discrete(action_cfg.ACTION_DIM_DISCRETE_KEYBOARD),
        })

        # Initialize components
        print("Initializing LoL Environment...")

        # Screen capture
        self.capture = ScreenCapture(target_fps=capture_cfg.TARGET_FPS)

        # Vision pipeline
        self.vision = VisionPipeline(use_yolo=use_yolo)

        # Input controller (only if not headless)
        self.input_controller = None
        if not headless:
            try:
                self.input_controller = InputController()
            except Exception as e:
                print(f"Warning: Could not initialize input controller: {e}")
                print("Running in observation-only mode")

        # CS detector for reward calculation
        # Prefer Live Client API over OCR (faster, more accurate)
        self.use_live_client = False
        if LIVE_CLIENT_AVAILABLE:
            try:
                self.cs_detector = LiveClientCSDetector()
                # Check if API is actually available (game running)
                if LiveClientAPI().is_game_running():
                    self.use_live_client = True
                    print("Using Live Client API for CS detection (fast & accurate)")
                else:
                    print("Live Client API not available (no game running), falling back to OCR")
                    self.cs_detector = CSDetector(
                        screen_width=capture_cfg.SCREEN_WIDTH,
                        screen_height=capture_cfg.SCREEN_HEIGHT,
                        update_interval=0.5
                    )
            except Exception as e:
                print(f"Live Client API failed: {e}, falling back to OCR")
                self.cs_detector = CSDetector(
                    screen_width=capture_cfg.SCREEN_WIDTH,
                    screen_height=capture_cfg.SCREEN_HEIGHT,
                    update_interval=0.5
                )
        else:
            self.cs_detector = CSDetector(
                screen_width=capture_cfg.SCREEN_WIDTH,
                screen_height=capture_cfg.SCREEN_HEIGHT,
                update_interval=0.5  # Update every 0.5 seconds
            )
        self.last_cs_reward_cs = 0  # Track CS for reward calculation

        # Shaping reward tracking
        self.last_minion_distance = None  # Track distance to nearest minion
        self.last_action = None  # Track last action for reward shaping
        self.mouse_x = capture_cfg.SCREEN_WIDTH // 2
        self.mouse_y = capture_cfg.SCREEN_HEIGHT // 2

        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_start_time = 0.0

        # Game state tracking
        self.previous_game_state: Optional[GameState] = None
        self.current_game_state: Optional[GameState] = None

        # Curriculum learning tracking
        self.total_cs = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.total_objectives = 0

        # Start kill switch listener
        _start_kill_switch_listener()

        print("Environment initialized successfully")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()

        # Reset game state
        self.previous_game_state = None
        self.current_game_state = None

        # Reset curriculum metrics
        self.total_cs = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.total_objectives = 0

        # Reset CS detector
        self.cs_detector.reset()
        self.current_cs = 0
        self.cs_gained = 0

        # Reset shaping reward tracking
        self.last_minion_distance = None
        self.last_action = None
        self.mouse_x = capture_cfg.SCREEN_WIDTH // 2
        self.mouse_y = capture_cfg.SCREEN_HEIGHT // 2

        # Start screen capture if not already running
        if not self.capture.running:
            self.capture.start()
            time.sleep(0.5)  # Allow capture to stabilize

        # Get initial observation
        observation = self._get_observation()

        info = {
            "episode": 0,
            "curriculum_stage": self.curriculum_stage,
        }

        return observation, info

    def step(
        self,
        action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary with continuous and discrete actions

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.episode_steps += 1

        # Track action for reward shaping
        self.last_action = action

        # Update estimated mouse position from action
        if "continuous" in action:
            continuous = action["continuous"]
            # Mouse moves up to 200 pixels per action
            self.mouse_x = int(np.clip(
                self.mouse_x + continuous[0] * 200,
                0, capture_cfg.SCREEN_WIDTH - 1
            ))
            self.mouse_y = int(np.clip(
                self.mouse_y + continuous[1] * 200,
                0, capture_cfg.SCREEN_HEIGHT - 1
            ))

        # Execute action
        if self.input_controller is not None and not self.headless:
            try:
                self.input_controller.execute_action(action)
            except Exception as e:
                print(f"Warning: Action execution failed: {e}")

        # Small delay to allow game to respond
        time.sleep(0.016)  # ~60 FPS

        # Get new observation
        observation = self._get_observation()

        # Update CS detector
        self.current_cs, self.cs_gained = self.cs_detector.update()

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Build info dict
        info = self._build_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from vision pipeline"""
        # Get latest frame from capture
        frame_data = self.capture.get_latest_frame()

        if frame_data is None:
            # Return zero observation if no frame available
            return np.zeros(vision_cfg.STATE_DIM, dtype=np.float32)

        # Process frame through vision pipeline
        self.previous_game_state = self.current_game_state
        self.current_game_state = self.vision.process_frame(frame_data)

        return self.current_game_state.feature_vector

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current curriculum stage.

        Implements curriculum learning with different reward functions:
        - cs_training: Focus on last-hitting minions
        - trading: Focus on damage dealt/taken
        - objectives: Focus on towers/dragons/baron
        """
        if self.current_game_state is None:
            return 0.0

        reward = 0.0

        # Stage-specific rewards
        if self.curriculum_stage == "cs_training":
            reward += self._reward_cs_training()
        elif self.curriculum_stage == "trading":
            reward += self._reward_trading()
        elif self.curriculum_stage == "objectives":
            reward += self._reward_objectives()

        # Global penalties
        reward += self._global_penalties()

        return reward

    def _reward_cs_training(self) -> float:
        """Reward function for CS training stage with shaping rewards"""
        reward = 0.0

        # =======================================================================
        # PRIMARY REWARD: CS gained (detected via OCR)
        # This is the main goal - everything else is shaping to help discover it
        # =======================================================================
        if self.cs_gained > 0:
            reward += reward_cfg.REWARD_CS_HIT * self.cs_gained
            self.total_cs += self.cs_gained
            print(f"  +++ CS REWARD: +{self.cs_gained} CS (total: {self.total_cs}) reward: +{reward_cfg.REWARD_CS_HIT * self.cs_gained:.2f}")

        # =======================================================================
        # SHAPING REWARDS: Guide agent toward farming behavior
        # =======================================================================
        shaping_reward = 0.0

        # 1. Lane presence reward - be in the center area of screen
        shaping_reward += self._reward_lane_presence()

        # 2. Minion proximity rewards (if we have detections)
        shaping_reward += self._reward_minion_proximity()

        # 3. Attack action reward - encourage right-clicking
        shaping_reward += self._reward_attack_action()

        # 4. Screen center reward - stay in playable area
        shaping_reward += self._reward_screen_position()

        # 5. Camera lock reward - keep camera locked (penalize Y key)
        shaping_reward += self._reward_camera_lock()

        # 6. Top lane positioning - Garen should stay in top lane
        shaping_reward += self._reward_top_lane_position()

        # 7. Shop penalty - don't click on shop or stay in base
        shaping_reward += self._reward_no_shop()

        # Add shaping to total (don't print every time - too spammy)
        reward += shaping_reward

        # Small penalty for being completely idle
        if self.input_controller and self.input_controller.get_current_apm() < 10:
            reward += reward_cfg.PENALTY_IDLE_PER_SECOND * 0.01

        return reward

    def _reward_lane_presence(self) -> float:
        """Reward for being in the lane area (center of screen)"""
        # In LoL, during laning, the action happens in the center of the screen
        # Reward being in the vertical center (not too far up or down)
        screen_center_y = capture_cfg.SCREEN_HEIGHT // 2
        y_distance = abs(self.mouse_y - screen_center_y) / (capture_cfg.SCREEN_HEIGHT // 2)

        # More reward for being closer to center
        if y_distance < 0.5:  # Within center half of screen
            return reward_cfg.REWARD_LANE_PRESENCE
        return 0.0

    def _reward_minion_proximity(self) -> float:
        """Reward for being near minions and approaching them"""
        reward = 0.0

        if self.current_game_state is None:
            return reward

        # Find minion detections
        minions = self._get_minion_detections()

        if not minions:
            # No minions detected - give small reward for being in lane area anyway
            # This helps when YOLO doesn't detect LoL-specific objects
            return 0.0

        # Calculate distance to nearest minion
        nearest_distance = float('inf')
        nearest_minion = None

        player_x = capture_cfg.SCREEN_WIDTH // 2  # Assume player at center
        player_y = capture_cfg.SCREEN_HEIGHT // 2

        for minion in minions:
            center_x, center_y = minion.center
            # Normalized distance
            dist = np.sqrt(
                ((center_x - player_x) / capture_cfg.SCREEN_WIDTH) ** 2 +
                ((center_y - player_y) / capture_cfg.SCREEN_HEIGHT) ** 2
            )
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_minion = minion

        # Reward for being close to minions
        if nearest_distance < reward_cfg.MINION_PROXIMITY_THRESHOLD:
            reward += reward_cfg.REWARD_NEAR_MINIONS

        # Reward for cursor being near minion
        if nearest_minion:
            cursor_dist = np.sqrt(
                ((self.mouse_x - nearest_minion.center[0]) / capture_cfg.SCREEN_WIDTH) ** 2 +
                ((self.mouse_y - nearest_minion.center[1]) / capture_cfg.SCREEN_HEIGHT) ** 2
            )
            if cursor_dist < reward_cfg.CURSOR_TARGET_THRESHOLD:
                reward += reward_cfg.REWARD_CURSOR_NEAR_MINION

        # Reward for moving closer to minions (approach reward)
        if self.last_minion_distance is not None and nearest_distance < self.last_minion_distance:
            reward += reward_cfg.REWARD_APPROACH_MINION

        self.last_minion_distance = nearest_distance

        return reward

    def _reward_attack_action(self) -> float:
        """Reward for taking attack actions (right-click)"""
        if self.last_action is None:
            return 0.0

        # Right-click is action 2 in discrete_mouse
        discrete_mouse = self.last_action.get("discrete_mouse", 0)
        if discrete_mouse == 2:  # Right click
            return reward_cfg.REWARD_ATTACK_ACTION

        return 0.0

    def _reward_screen_position(self) -> float:
        """Reward for keeping mouse in the center playable area"""
        # Reward being in the center 60% of the screen
        x_norm = abs(self.mouse_x - capture_cfg.SCREEN_WIDTH // 2) / (capture_cfg.SCREEN_WIDTH // 2)
        y_norm = abs(self.mouse_y - capture_cfg.SCREEN_HEIGHT // 2) / (capture_cfg.SCREEN_HEIGHT // 2)

        if x_norm < 0.6 and y_norm < 0.6:
            return reward_cfg.REWARD_SCREEN_CENTER
        return 0.0

    def _reward_camera_lock(self) -> float:
        """
        Reward for keeping camera locked.
        Since Y key is not in the action space, agent can't unlock camera.
        Just give a small constant reward as long as we assume camera is locked.
        """
        # Y key is not in KEYBOARD_KEYS, so agent cannot press it
        # Give small reward for having camera locked (which should always be true)
        return reward_cfg.REWARD_CAMERA_LOCKED

    def _reward_top_lane_position(self) -> float:
        """
        Reward for being in top lane (Garen's lane).
        Detects position from minimap - top lane is in top-left area.
        """
        reward = 0.0

        # Get minimap from capture
        frame_data = self.capture.get_latest_frame()
        if frame_data is None or "minimap" not in frame_data.regions:
            return reward

        minimap = frame_data.regions["minimap"]
        if minimap is None:
            return reward

        # Analyze minimap to find player position
        # Player icon is typically a bright circle/arrow
        # In top lane, player should be in top-left quadrant of minimap
        player_pos = self._find_player_on_minimap(minimap)

        if player_pos is not None:
            px, py = player_pos
            minimap_h, minimap_w = minimap.shape[:2]

            # Normalize position (0-1 scale)
            px_norm = px / minimap_w
            py_norm = py / minimap_h

            # Top lane is roughly:
            # - X: 0.0 to 0.5 (left half) or along the diagonal
            # - Y: 0.0 to 0.5 (top half)
            # The lane runs diagonally from top-left to somewhere in the middle

            # Check if in top lane area (top-left quadrant + diagonal)
            in_top_lane = False

            # Top-left quadrant check
            if px_norm < 0.4 and py_norm < 0.4:
                in_top_lane = True
            # Diagonal lane check (top lane runs from top-left toward center)
            elif px_norm < 0.6 and py_norm < 0.6 and (px_norm + py_norm) < 0.8:
                in_top_lane = True

            if in_top_lane:
                reward += reward_cfg.REWARD_TOP_LANE
            else:
                # Penalize being in wrong area (mid, bot, jungle)
                reward += reward_cfg.PENALTY_WRONG_LANE

        return reward

    def _reward_no_shop(self) -> float:
        """
        Penalize clicking on shop area or being in base.
        Shop icon is in bottom-right corner near minimap.
        """
        reward = 0.0

        # Check if last action was a click in the shop area
        if self.last_action is not None:
            discrete_mouse = self.last_action.get("discrete_mouse", 0)

            # If clicked (left or right click)
            if discrete_mouse in [1, 2]:
                # Shop icon is in bottom-right area of screen
                # For 1920x1080: roughly x > 1750, y > 900
                shop_x_threshold = capture_cfg.SCREEN_WIDTH * 0.91  # Right 9% of screen
                shop_y_threshold = capture_cfg.SCREEN_HEIGHT * 0.83  # Bottom 17% of screen

                if self.mouse_x > shop_x_threshold and self.mouse_y > shop_y_threshold:
                    reward += reward_cfg.PENALTY_SHOP_CLICK

                # Also penalize clicking on bottom HUD area (items, abilities area excluded)
                # Items are in bottom-center, but shop button is bottom-right
                gold_area_x_min = capture_cfg.SCREEN_WIDTH * 0.85
                gold_area_y_min = capture_cfg.SCREEN_HEIGHT * 0.90

                if self.mouse_x > gold_area_x_min and self.mouse_y > gold_area_y_min:
                    reward += reward_cfg.PENALTY_SHOP_CLICK * 0.5  # Half penalty for gold area

        # Also penalize being in base (detected from minimap position)
        # Base is bottom-left corner for blue team
        frame_data = self.capture.get_latest_frame()
        if frame_data is not None and "minimap" in frame_data.regions:
            minimap = frame_data.regions["minimap"]
            player_pos = self._find_player_on_minimap(minimap)

            if player_pos is not None:
                px, py = player_pos
                minimap_h, minimap_w = minimap.shape[:2]
                px_norm = px / minimap_w
                py_norm = py / minimap_h

                # Base/fountain is in bottom-left corner (x < 0.15, y > 0.85)
                if px_norm < 0.15 and py_norm > 0.85:
                    reward += reward_cfg.PENALTY_IN_BASE

        return reward

    def _find_player_on_minimap(self, minimap: np.ndarray) -> Optional[tuple]:
        """
        Find player position on minimap.
        Returns (x, y) coordinates or None if not found.
        """
        try:
            # Player icon is typically bright (white/yellow/green depending on team)
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

            # Look for bright spots (player arrow/circle is usually very bright)
            # The player's own icon is typically cyan/teal for blue team or red for red team
            # But it's always the brightest element in its area

            # Method 1: Find brightest point
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            # Apply slight blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Find the maximum brightness point
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

            # The brightest point is likely the player icon
            if max_val > 200:  # Threshold for "bright"
                return max_loc  # Returns (x, y)

            # Method 2: Look for cyan/teal color (blue team player icon)
            # Cyan in HSV: H=80-100, S=100-255, V=200-255
            lower_cyan = np.array([80, 100, 200])
            upper_cyan = np.array([100, 255, 255])
            mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

            # Find centroid of cyan areas
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return (cx, cy)

            return None

        except Exception:
            return None

    def _get_minion_detections(self):
        """Get minion-like detections from current game state"""
        if self.current_game_state is None:
            return []

        minions = []
        minion_classes = {"minion_melee", "minion_ranged", "minion_cannon", "minion_super"}

        for detection in self.current_game_state.detections:
            # Check if it's a minion class
            if detection.class_name in minion_classes:
                minions.append(detection)
            # Also accept generic "person" detections as potential targets
            # (generic YOLO might detect minions as various objects)
            elif detection.class_name in {"person", "bird", "cat", "dog"}:
                # Generic objects that might be game entities
                minions.append(detection)

        return minions

    def _reward_trading(self) -> float:
        """Reward function for trading stage"""
        reward = 0.0

        # Detect damage dealt/taken by monitoring HP bars
        if self.previous_game_state and self.current_game_state:
            hp_change = (
                self.current_game_state.player_hp_percent -
                self.previous_game_state.player_hp_percent
            )

            if hp_change < 0:
                # Took damage
                reward += reward_cfg.PENALTY_DAMAGE_TAKEN * abs(hp_change) * 1000

            # Detect kills/deaths would require game API

        return reward

    def _reward_objectives(self) -> float:
        """Reward function for objectives stage"""
        reward = 0.0

        # Objective rewards would require integration with Riot API
        # to detect tower destructions, dragon kills, etc.

        return reward

    def _global_penalties(self) -> float:
        """Apply global penalties (AFK detection, etc.)"""
        penalty = 0.0

        # Penalize very low APM (possible AFK)
        if self.input_controller:
            apm = self.input_controller.get_current_apm()
            if apm < 10:
                penalty += reward_cfg.PENALTY_AFK_DETECTED * 0.1

        return penalty

    def _is_terminated(self) -> bool:
        """Check if episode should terminate (game ended or kill switch pressed)"""
        # Check kill switch
        if is_kill_switch_pressed():
            return True

        # This would require checking game state from Riot API
        # For now, just return False
        return False

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit, etc.)"""
        # Check maximum episode length
        episode_duration = time.time() - self.episode_start_time
        max_duration = safety_cfg.MAX_GAME_DURATION_MINUTES * 60

        if episode_duration > max_duration:
            return True

        return False

    def _build_info(self) -> Dict[str, Any]:
        """Build info dictionary with episode statistics"""
        info = {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "curriculum_stage": self.curriculum_stage,
        }

        # Add performance metrics
        if self.capture:
            info.update(self.capture.get_performance_stats())

        if self.vision:
            info.update(self.vision.get_performance_stats())

        if self.input_controller:
            info["apm"] = self.input_controller.get_current_apm()

        # Add game state info
        if self.current_game_state:
            info["player_hp"] = self.current_game_state.player_hp_percent
            info["player_mana"] = self.current_game_state.player_mana_percent
            info["detections"] = len(self.current_game_state.detections)

        # Add CS stats
        if self.cs_detector:
            cs_stats = self.cs_detector.get_stats()
            info["cs"] = cs_stats["current_cs"]
            info["cs_per_minute"] = cs_stats["cs_per_minute"]
            info["total_cs"] = self.total_cs

        return info

    def render(self):
        """Render the environment (return current screen)"""
        if self.render_mode == "rgb_array":
            frame_data = self.capture.get_latest_frame()
            if frame_data and "main" in frame_data.regions:
                return frame_data.regions["main"]
        return None

    def close(self):
        """Clean up resources"""
        print("Closing environment...")

        if self.capture:
            self.capture.stop()

        # Stop kill switch listener
        global _kill_switch_listener
        if _kill_switch_listener:
            _kill_switch_listener.stop()
            _kill_switch_listener = None

        print("Environment closed")


class LoLPracticeTool(LoLEnvironment):
    """
    Specialized environment for Practice Tool training.
    Assumes simplified game state and focuses on mechanical practice.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.practice_mode = True

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with practice tool specific initialization"""
        obs, info = super().reset(**kwargs)
        info["practice_mode"] = True
        return obs, info


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Testing LoL Environment")
    print("=" * 60)

    # Create environment (headless mode for testing without game running)
    env = LoLEnvironment(
        render_mode="rgb_array",
        curriculum_stage="cs_training",
        use_yolo=False,  # Disable YOLO for quick testing
        headless=True  # No input execution
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Info: {info}")

    # Run a few steps with random actions
    print("\nRunning 10 random steps...")
    for i in range(10):
        # Sample random action
        action = {
            "continuous": env.action_space["continuous"].sample(),
            "discrete_mouse": env.action_space["discrete_mouse"].sample(),
            "discrete_keyboard": env.action_space["discrete_keyboard"].sample(),
        }

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        print(f"  Info: FPS={info.get('fps', 0):.1f}, APM={info.get('apm', 0):.1f}")

        if terminated or truncated:
            print("Episode ended")
            break

        time.sleep(0.1)

    # Clean up
    env.close()

    print("\nEnvironment testing completed!")
