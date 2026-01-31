"""
Sim-to-Real Training Script for LoL CS Training.

Training Pipeline:
1. Pre-train in simulation (fast, safe, domain randomized)
2. Fine-tune on real game (slower, uses actual game)

Usage:
    # Train in simulation only
    python train_sim2real.py --sim-only --steps 500000

    # Train in sim, then transfer to real game
    python train_sim2real.py --steps 500000 --finetune-steps 50000

    # Resume from checkpoint
    python train_sim2real.py --resume checkpoints/sim_model.zip
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3[extra]")

# Import environments
from src.lol_sim_aligned import LoLSimAligned

try:
    from src.lol_env import LoLEnvironment
    REAL_ENV_AVAILABLE = True
except ImportError:
    REAL_ENV_AVAILABLE = False
    print("Warning: Real environment not available (missing dependencies)")


def make_sim_env(rank: int, seed: int = 0, domain_randomization: bool = True):
    """Factory function for creating simulation environments."""
    def _init():
        env = LoLSimAligned(
            render_mode=None,
            domain_randomization=domain_randomization,
            simulation_speed=2.0,  # 2x speed for faster training
            max_episode_steps=3000,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_real_env(seed: int = 0):
    """Factory function for creating real game environment."""
    def _init():
        env = LoLEnvironment(
            render_mode=None,
            curriculum_stage="cs_training",
            use_yolo=False,  # Disable YOLO for speed, use Live Client API
            headless=False,
        )
        return env
    return _init


def train_simulation(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_dir: str = "checkpoints/sim",
    log_dir: str = "logs/sim",
    resume_path: Optional[str] = None,
    domain_randomization: bool = True,
):
    """
    Train agent in simulation environment.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
        resume_path: Path to resume training from
        domain_randomization: Enable domain randomization

    Returns:
        Trained model
    """
    print("=" * 60)
    print("SIMULATION TRAINING")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Domain randomization: {domain_randomization}")
    print()

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment
    if n_envs > 1:
        env = SubprocVecEnv([
            make_sim_env(i, seed=42, domain_randomization=domain_randomization)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_sim_env(0, seed=42, domain_randomization=domain_randomization)])

    env = VecMonitor(env, log_dir)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_sim_env(100, seed=100, domain_randomization=False)])

    # Create or load model
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="sim_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(20000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train
    print("\nStarting training...")
    print("Monitor with: tensorboard --logdir", log_dir)
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = os.path.join(save_dir, "sim_model_final.zip")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    env.close()
    eval_env.close()

    return model


def finetune_real(
    model,
    total_timesteps: int = 50_000,
    save_dir: str = "checkpoints/real",
    log_dir: str = "logs/real",
):
    """
    Fine-tune pre-trained model on real game environment.

    Args:
        model: Pre-trained model from simulation
        total_timesteps: Fine-tuning steps
        save_dir: Directory to save checkpoints
        log_dir: Directory for logs

    Returns:
        Fine-tuned model
    """
    if not REAL_ENV_AVAILABLE:
        print("ERROR: Real environment not available")
        print("Make sure League of Legends is running and dependencies are installed")
        return model

    print("=" * 60)
    print("REAL GAME FINE-TUNING")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print()
    print("IMPORTANT:")
    print("  1. Make sure League of Legends is running")
    print("  2. Enter Practice Tool with Garen")
    print("  3. Press F12 to stop training anytime")
    print()

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create real environment (single env only)
    env = DummyVecEnv([make_real_env()])
    env = VecMonitor(env, log_dir)

    # Update model's environment
    model.set_env(env)

    # Lower learning rate for fine-tuning
    model.learning_rate = 1e-4

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir,
        name_prefix="real_model",
    )

    callbacks = CallbackList([checkpoint_callback])

    # Train
    print("\nStarting fine-tuning on real game...")
    print("Press F12 to stop")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,  # Continue counting from sim training
        )
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user")

    # Save final model
    final_path = os.path.join(save_dir, "real_model_final.zip")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    env.close()

    return model


def evaluate_model(model, env_type: str = "sim", episodes: int = 5, render: bool = True):
    """
    Evaluate a trained model.

    Args:
        model: Trained model
        env_type: "sim" or "real"
        episodes: Number of evaluation episodes
        render: Whether to render
    """
    print("=" * 60)
    print(f"EVALUATING MODEL ({env_type})")
    print("=" * 60)

    if env_type == "sim":
        env = LoLSimAligned(
            render_mode="human" if render else None,
            domain_randomization=False,
            simulation_speed=1.0,
        )
    else:
        if not REAL_ENV_AVAILABLE:
            print("Real environment not available")
            return
        env = LoLEnvironment(
            render_mode="rgb_array" if render else None,
            curriculum_stage="cs_training",
            use_yolo=False,
            headless=False,
        )

    results = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if render and env_type == "sim":
                env.render()

        results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps,
            "cs": info.get("last_hits", info.get("cs", 0)),
            "gold": info.get("gold", info.get("total_gold", 0)),
        })

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, CS={results[-1]['cs']}, "
              f"gold={results[-1]['gold']}")

    env.close()

    # Summary
    avg_reward = np.mean([r["reward"] for r in results])
    avg_cs = np.mean([r["cs"] for r in results])
    print(f"\nAverage reward: {avg_reward:.2f}")
    print(f"Average CS: {avg_cs:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real LoL Training")

    parser.add_argument("--steps", type=int, default=500_000,
                       help="Simulation training steps")
    parser.add_argument("--finetune-steps", type=int, default=0,
                       help="Real game fine-tuning steps (0 to skip)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel simulation environments")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model checkpoint to resume from")
    parser.add_argument("--sim-only", action="store_true",
                       help="Only train in simulation, skip real game")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate a trained model")
    parser.add_argument("--no-domain-rand", action="store_true",
                       help="Disable domain randomization")
    parser.add_argument("--eval-episodes", type=int, default=5,
                       help="Number of evaluation episodes")

    args = parser.parse_args()

    if not SB3_AVAILABLE:
        print("Please install stable-baselines3:")
        print("  pip install stable-baselines3[extra]")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_sim = f"checkpoints/sim_{timestamp}"
    save_dir_real = f"checkpoints/real_{timestamp}"
    log_dir_sim = f"logs/sim_{timestamp}"
    log_dir_real = f"logs/real_{timestamp}"

    if args.eval_only:
        if not args.resume:
            print("ERROR: --resume required for evaluation")
            sys.exit(1)
        model = PPO.load(args.resume)
        evaluate_model(model, env_type="sim", episodes=args.eval_episodes)
        return

    # Phase 1: Simulation training
    model = train_simulation(
        total_timesteps=args.steps,
        n_envs=args.n_envs,
        save_dir=save_dir_sim,
        log_dir=log_dir_sim,
        resume_path=args.resume,
        domain_randomization=not args.no_domain_rand,
    )

    # Evaluate in sim
    print("\n")
    evaluate_model(model, env_type="sim", episodes=3, render=False)

    # Phase 2: Real game fine-tuning
    if not args.sim_only and args.finetune_steps > 0:
        print("\n")
        model = finetune_real(
            model,
            total_timesteps=args.finetune_steps,
            save_dir=save_dir_real,
            log_dir=log_dir_real,
        )

        # Evaluate on real
        evaluate_model(model, env_type="real", episodes=3, render=True)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Simulation model: {save_dir_sim}/sim_model_final.zip")
    if not args.sim_only and args.finetune_steps > 0:
        print(f"Real game model: {save_dir_real}/real_model_final.zip")


if __name__ == "__main__":
    main()
