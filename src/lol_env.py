"""
League of Legends Gymnasium Environment
Wraps the capture, vision, and input systems into a standard RL environment.
"""
import time
import threading
from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import vision_cfg, action_cfg, reward_cfg, safety_cfg, capture_cfg
from .capture import ScreenCapture, FrameData
from .vision import VisionPipeline, GameState
from .input_controller import InputController

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
        """Reward function for CS training stage"""
        reward = 0.0

        # Detect CS hits by checking for gold increase
        # (This is simplified - real implementation would use game API)

        # Reward for staying active (not idle)
        if self.input_controller and self.input_controller.get_current_apm() > 20:
            reward += 0.01
        else:
            reward += reward_cfg.PENALTY_IDLE_PER_SECOND * 0.1

        # Placeholder: Detect level up
        # Real implementation would check game state from Riot API

        return reward

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
