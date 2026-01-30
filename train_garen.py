#!/usr/bin/env python3
"""
Garen-specific training script with custom reward shaping and curriculum.
"""
import argparse
import time
from pathlib import Path

from train import Trainer, RolloutBuffer
from src.lol_env import LoLPracticeTool
from src.rl_agent import PPOAgent
from src.garen_config import (
    garen_rewards,
    garen_abilities,
    garen_combos,
    GAREN_CURRICULUM,
    get_garen_reward
)


class GarenEnvironment(LoLPracticeTool):
    """
    Customized environment for Garen training with specific reward shaping.
    """

    def __init__(self, curriculum_stage: str = "stage_1_farming", **kwargs):
        super().__init__(**kwargs)
        self.garen_stage = curriculum_stage
        self.stage_config = GAREN_CURRICULUM[curriculum_stage]

        print(f"\n{'='*60}")
        print(f"Garen Training Stage: {curriculum_stage}")
        print(f"Focus: {self.stage_config['focus']}")
        print(f"Success Metric: {self.stage_config['success_metric']}")
        print(f"{'='*60}\n")

    def _calculate_reward(self) -> float:
        """Override with Garen-specific reward calculation"""
        reward = super()._calculate_reward()

        # Add Garen-specific rewards
        if self.current_game_state and self.previous_game_state:
            # Get the last action taken
            last_action = getattr(self, 'last_action', {})

            # Calculate Garen-specific reward
            garen_bonus = get_garen_reward(
                game_state=self._state_to_dict(self.current_game_state),
                action=last_action,
                previous_state=self._state_to_dict(self.previous_game_state)
            )

            reward += garen_bonus

        return reward

    def step(self, action):
        """Override to track actions for reward calculation"""
        self.last_action = action
        return super().step(action)

    def _state_to_dict(self, game_state) -> dict:
        """Convert GameState to dict for reward calculation"""
        if game_state is None:
            return {}

        return {
            "hp": game_state.player_hp_percent,
            "mana": game_state.player_mana_percent,
            "cooldowns": game_state.ability_cooldowns,
            "detections": len(game_state.detections)
        }


class GarenTrainer(Trainer):
    """Extended trainer with Garen-specific logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.garen_stats = {
            "q_uses": 0,
            "e_uses": 0,
            "r_uses": 0,
            "successful_combos": 0
        }

    def _log_episode_stats(self):
        """Override to include Garen-specific stats"""
        super()._log_episode_stats()

        print("Garen Stats:")
        print(f"  Q uses: {self.garen_stats['q_uses']}")
        print(f"  E uses: {self.garen_stats['e_uses']}")
        print(f"  R uses: {self.garen_stats['r_uses']}")
        print(f"  Successful combos: {self.garen_stats['successful_combos']}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoL RL Agent on Garen")

    parser.add_argument(
        "--stage",
        type=str,
        default="stage_1_farming",
        choices=list(GAREN_CURRICULUM.keys()),
        help="Garen training curriculum stage"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no input execution)"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/garen",
        help="Directory for saving Garen checkpoints"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo to test setup (10 seconds)"
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for testing"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🛡️  GAREN RL TRAINING")
    print("=" * 80)
    print()

    # Get stage config
    stage_config = GAREN_CURRICULUM[args.stage]

    print("Champion: Garen")
    print(f"Stage: {args.stage}")
    print(f"Focus: {stage_config['focus']}")
    print(f"Success Metric: {stage_config['success_metric']}")
    print(f"Training Steps: {stage_config['duration_steps']:,}")
    print()

    # Print Garen ability info
    print("Garen Abilities:")
    print(f"  Q (Decisive Strike): {garen_abilities.Q_COOLDOWN}s cooldown")
    print(f"  W (Courage): {garen_abilities.W_COOLDOWN}s cooldown")
    print(f"  E (Judgment): {garen_abilities.E_COOLDOWN}s cooldown, {garen_abilities.E_DURATION}s duration")
    print(f"  R (Demacian Justice): {garen_abilities.R_COOLDOWN_EARLY}s cooldown")
    print()

    # Safety warning
    if not args.headless and not args.demo:
        print("⚠️  WARNING: Input execution enabled!")
        print("1. Open League of Legends")
        print("2. Start a Practice Tool game")
        print("3. Pick GAREN")
        print("4. Wait until you're in-game")
        print()
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return

    # Create Garen environment
    env = GarenEnvironment(
        curriculum_stage=args.stage,
        headless=args.headless,
        use_yolo=True
    )

    # Create agent
    agent = PPOAgent()

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)

    # Demo mode (quick test)
    if args.demo:
        print("\n🎮 Running 10-second demo...")
        obs, info = env.reset()
        agent.reset_hidden_state()

        start_time = time.time()
        while time.time() - start_time < 10.0:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            time.sleep(0.1)

        env.close()
        print("✓ Demo completed successfully!")
        print("\nNext step: Run full training:")
        print(f"  python train_garen.py --stage {args.stage}")
        return

    # Create trainer
    total_steps = args.total_timesteps if args.total_timesteps else stage_config['duration_steps']
    trainer = GarenTrainer(
        env=env,
        agent=agent,
        total_timesteps=total_steps,
        checkpoint_dir=args.checkpoint_dir
    )

    # Start training
    print("\n🚀 Starting Garen training...")
    print("Press Ctrl+C to stop and save progress\n")

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

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("\nNext steps:")
    print(f"1. Check training results in dashboard")
    print(f"2. Test the trained model:")
    print(f"   python main.py infer --model {args.checkpoint_dir}/best_model.pt --practice-tool")
    print(f"3. Progress to next stage:")

    # Suggest next stage
    stages = list(GAREN_CURRICULUM.keys())
    current_idx = stages.index(args.stage)
    if current_idx < len(stages) - 1:
        next_stage = stages[current_idx + 1]
        print(f"   python train_garen.py --stage {next_stage}")
    else:
        print("   🎉 You've completed all training stages!")

    print("=" * 80)


if __name__ == "__main__":
    main()
