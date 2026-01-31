"""
Training Script for LoL RL Agent
Implements the main training loop with curriculum learning and checkpointing.
"""
import os
import re
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.config import rl_cfg, logging_cfg, safety_cfg
from src.lol_env import LoLEnvironment, LoLPracticeTool, is_kill_switch_pressed
from src.rl_agent import PPOAgent


class RolloutBuffer:
    """
    Buffer for storing rollout data before training updates.
    """

    def __init__(self, buffer_size: int = rl_cfg.PPO_N_STEPS):
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        """Clear the buffer"""
        self.states = []
        self.continuous_actions = []
        self.discrete_mouse_actions = []
        self.discrete_keyboard_actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []

    def add(
        self,
        state: np.ndarray,
        action_dict: Dict[str, Any],
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.continuous_actions.append(action_dict["continuous"])
        self.discrete_mouse_actions.append(action_dict["discrete_mouse"])
        self.discrete_keyboard_actions.append(action_dict["discrete_keyboard"])
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.states) >= self.buffer_size

    def get(self) -> Dict[str, np.ndarray]:
        """Get buffer contents as arrays"""
        return {
            "states": np.array(self.states),
            "continuous_actions": np.array(self.continuous_actions),
            "discrete_mouse_actions": np.array(self.discrete_mouse_actions),
            "discrete_keyboard_actions": np.array(self.discrete_keyboard_actions),
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "dones": np.array(self.dones),
            "log_probs": np.array(self.log_probs),
        }

    def __len__(self):
        return len(self.states)


class Trainer:
    """
    Main training class that orchestrates the training loop.
    """

    def __init__(
        self,
        env: LoLEnvironment,
        agent: PPOAgent,
        total_timesteps: int = rl_cfg.TOTAL_TIMESTEPS,
        checkpoint_dir: str = logging_cfg.CHECKPOINT_DIR,
        log_interval: int = rl_cfg.LOG_INTERVAL,
    ):
        self.env = env
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.timesteps_done = 0
        self.episodes_done = 0
        self.best_reward = -float("inf")

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []

        # Performance tracking
        self.training_start_time = time.time()
        self.last_progress_print = time.time()
        self.progress_print_interval = 30.0  # Print progress every 30 seconds

    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print()

        # Reset environment
        obs, info = self.env.reset()
        self.agent.reset_hidden_state()

        episode_reward = 0.0
        episode_length = 0

        while self.timesteps_done < self.total_timesteps:
            # Collect rollout
            while not self.rollout_buffer.is_full():
                # Select action
                action_dict = self.agent.select_action(obs)

                # Execute action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action_dict)

                # Store transition (simplified - need proper log_prob calculation)
                self.rollout_buffer.add(
                    state=obs,
                    action_dict=action_dict,
                    reward=reward,
                    done=terminated or truncated,
                    value=action_dict["value"],
                    log_prob=0.0  # Placeholder - would need to compute from policy
                )

                # Update tracking
                episode_reward += reward
                episode_length += 1
                self.timesteps_done += 1

                # Check kill switch
                if is_kill_switch_pressed():
                    print("\nKill switch detected - stopping training...")
                    self._save_checkpoint()
                    return

                # Print progress every N seconds (so user knows it's working)
                if time.time() - self.last_progress_print > self.progress_print_interval:
                    self._print_progress(episode_reward, episode_length)
                    self.last_progress_print = time.time()

                # Handle episode end
                if terminated or truncated:
                    self.episodes_done += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    # Log episode
                    if self.episodes_done % self.log_interval == 0:
                        self._log_episode_stats()

                    # Reset for next episode
                    obs, info = self.env.reset()
                    self.agent.reset_hidden_state()
                    episode_reward = 0.0
                    episode_length = 0
                else:
                    obs = next_obs

            # Compute advantages and returns
            rollout_data = self.rollout_buffer.get()
            next_value = self.agent.select_action(obs)["value"]

            advantages, returns = self.agent.compute_gae(
                rewards=rollout_data["rewards"],
                values=rollout_data["values"],
                dones=rollout_data["dones"],
                next_value=next_value
            )

            rollout_data["advantages"] = advantages
            rollout_data["returns"] = returns

            # Update policy
            train_metrics = self.agent.update(rollout_data)

            # Log training metrics
            if self.agent.n_updates % self.log_interval == 0:
                self._log_training_metrics(train_metrics)

            # Save checkpoint
            if self.timesteps_done % rl_cfg.CHECKPOINT_FREQ == 0:
                self._save_checkpoint()

            # Clear buffer
            self.rollout_buffer.reset()

        print("\nTraining completed!")
        self._save_checkpoint(final=True)

    def _log_episode_stats(self):
        """Log episode statistics"""
        recent_rewards = self.episode_rewards[-self.log_interval:]
        recent_lengths = self.episode_lengths[-self.log_interval:]

        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        max_reward = np.max(recent_rewards) if recent_rewards else 0
        min_reward = np.min(recent_rewards) if recent_rewards else 0

        elapsed_time = time.time() - self.training_start_time
        fps = self.timesteps_done / elapsed_time
        hours = int(elapsed_time // 3600)
        mins = int((elapsed_time % 3600) // 60)

        print()
        print("=" * 60)
        print(f"Episode {self.episodes_done} | Time: {hours}h {mins}m | Steps: {self.timesteps_done:,}/{self.total_timesteps:,}")
        print("=" * 60)
        print(f"  Avg Reward:  {avg_reward:.3f}  (min: {min_reward:.3f}, max: {max_reward:.3f})")
        print(f"  Avg Length:  {avg_length:.0f} steps")
        print(f"  FPS:         {fps:.1f}")
        print(f"  Best Reward: {self.best_reward:.3f}")

        # Show CS stats if available
        if hasattr(self.env, 'total_cs'):
            print(f"  Total CS:    {self.env.total_cs}")
        if hasattr(self.env, 'cs_detector'):
            cs_stats = self.env.cs_detector.get_stats()
            print(f"  CS/min:      {cs_stats['cs_per_minute']:.1f}")
            print(f"  OCR Acc:     {cs_stats['accuracy']:.1%}")

        print()

        # Track best reward
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            print(f"  *** New best reward! Saving checkpoint... ***")
            self._save_checkpoint(best=True)

    def _print_progress(self, current_episode_reward: float, current_episode_length: int):
        """Print periodic progress update during training"""
        elapsed = time.time() - self.training_start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        fps = self.timesteps_done / elapsed if elapsed > 0 else 0
        progress = (self.timesteps_done / self.total_timesteps) * 100

        print(f"\r[{hours:02d}:{mins:02d}:{secs:02d}] "
              f"Steps: {self.timesteps_done:,} ({progress:.1f}%) | "
              f"Episode: {self.episodes_done} | "
              f"Reward: {current_episode_reward:.2f} | "
              f"FPS: {fps:.1f}", end="", flush=True)

    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        print(f"\nTraining Update {self.agent.n_updates}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy_loss']:.4f}")
        print()

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint with training state"""
        if best:
            path = self.checkpoint_dir / "best_model.pt"
            print(f"Saving best model to {path}")
        elif final:
            path = self.checkpoint_dir / "final_model.pt"
            print(f"Saving final model to {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_{self.timesteps_done}.pt"
            print(f"Saving checkpoint to {path}")

        # Include training state so we can resume properly
        training_state = {
            "timesteps_done": self.timesteps_done,
            "episodes_done": self.episodes_done,
            "best_reward": self.best_reward,
        }
        self.agent.save(str(path), training_state=training_state)

    def load_training_state(self, training_state: dict):
        """Restore training state from checkpoint"""
        if training_state:
            self.timesteps_done = training_state.get("timesteps_done", 0)
            self.episodes_done = training_state.get("episodes_done", 0)
            self.best_reward = training_state.get("best_reward", -float("inf"))
            print(f"Restored training state: {self.timesteps_done:,} timesteps, {self.episodes_done} episodes")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LoL RL Agent")

    parser.add_argument(
        "--curriculum-stage",
        type=str,
        default="cs_training",
        choices=["cs_training", "trading", "objectives"],
        help="Curriculum learning stage"
    )

    parser.add_argument(
        "--practice-tool",
        action="store_true",
        help="Use Practice Tool environment"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no input execution)"
    )

    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO detection"
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=rl_cfg.TOTAL_TIMESTEPS,
        help="Total training timesteps"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=logging_cfg.CHECKPOINT_DIR,
        help="Directory for saving checkpoints"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint based on timestep number in filename."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    latest_checkpoint = None
    latest_timestep = -1

    for checkpoint_file in checkpoint_path.glob("checkpoint_*.pt"):
        # Extract timestep number from filename (e.g., checkpoint_4068.pt -> 4068)
        match = re.search(r"checkpoint_(\d+)\.pt$", checkpoint_file.name)
        if match:
            timestep = int(match.group(1))
            if timestep > latest_timestep:
                latest_timestep = timestep
                latest_checkpoint = str(checkpoint_file)

    return latest_checkpoint


def main():
    """Main training entry point"""
    args = parse_args()

    print("=" * 80)
    print("LoL RL Agent Training")
    print("=" * 80)
    print(f"Curriculum Stage: {args.curriculum_stage}")
    print(f"Practice Tool: {args.practice_tool}")
    print(f"Headless: {args.headless}")
    print(f"YOLO Enabled: {not args.no_yolo}")
    print()

    # Safety check
    if not args.headless and not args.practice_tool:
        print("WARNING: Running with input execution enabled!")
        print("Make sure you are in a Practice Tool or Custom game.")
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return

    # Create environment
    if args.practice_tool:
        env = LoLPracticeTool(
            curriculum_stage=args.curriculum_stage,
            use_yolo=not args.no_yolo,
            headless=args.headless
        )
    else:
        env = LoLEnvironment(
            curriculum_stage=args.curriculum_stage,
            use_yolo=not args.no_yolo,
            headless=args.headless
        )

    # Create agent
    agent = PPOAgent()

    # Resume from checkpoint - auto-detect latest if not specified
    resume_path = args.resume
    if resume_path is None:
        resume_path = find_latest_checkpoint(args.checkpoint_dir)
        if resume_path:
            print(f"Auto-detected latest checkpoint: {resume_path}")

    # Load checkpoint if available
    training_state = {}
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        training_state = agent.load(resume_path)

    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        total_timesteps=args.total_timesteps,
        checkpoint_dir=args.checkpoint_dir
    )

    # Restore training state (timesteps, episodes, best_reward)
    if training_state:
        trainer.load_training_state(training_state)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        trainer._save_checkpoint()
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Saving current model...")
        trainer._save_checkpoint()
    finally:
        env.close()


if __name__ == "__main__":
    main()
