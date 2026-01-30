#!/usr/bin/env python3
"""
Main entry point for LoL RL Agent
Provides a simple interface for running the agent in different modes.
"""
import argparse
import time
import sys

from src.config import safety_cfg
from src.lol_env import LoLEnvironment, LoLPracticeTool
from src.rl_agent import PPOAgent


def run_inference(
    model_path: str,
    practice_tool: bool = True,
    headless: bool = False,
    max_episodes: int = 1
):
    """
    Run trained agent in inference mode.

    Args:
        model_path: Path to trained model checkpoint
        practice_tool: Use Practice Tool environment
        headless: Run without input execution (observation only)
        max_episodes: Maximum number of episodes to run
    """
    print("=" * 80)
    print("LoL RL Agent - Inference Mode")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Practice Tool: {practice_tool}")
    print(f"Headless: {headless}")
    print(f"Kill switch: {safety_cfg.KILL_SWITCH_KEY}")
    print()

    # Safety check
    if not headless and not practice_tool:
        print("WARNING: Running with input execution enabled!")
        print("Make sure you are in a Practice Tool or Custom game.")
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return

    # Create environment
    if practice_tool:
        env = LoLPracticeTool(headless=headless, use_yolo=True)
    else:
        env = LoLEnvironment(headless=headless, use_yolo=True)

    # Create and load agent
    agent = PPOAgent()
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using random policy instead")

    # Run episodes
    try:
        for episode in range(max_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{max_episodes}")
            print('='*60)

            obs, info = env.reset()
            agent.reset_hidden_state()

            episode_reward = 0.0
            steps = 0

            done = False
            while not done:
                # Select action (deterministic for inference)
                action = agent.select_action(obs, deterministic=True)

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1

                # Print stats every 100 steps
                if steps % 100 == 0:
                    print(f"Step {steps}: Reward={episode_reward:.2f}, FPS={info.get('fps', 0):.1f}")

            print(f"\nEpisode finished!")
            print(f"Total steps: {steps}")
            print(f"Total reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        env.close()


def run_demo(headless: bool = True):
    """
    Run a simple demo showing the agent's capabilities.

    Args:
        headless: Run without input execution
    """
    print("=" * 80)
    print("LoL RL Agent - Demo Mode")
    print("=" * 80)
    print("This will demonstrate the agent's perception and decision-making")
    print(f"Headless mode: {headless}")
    print()

    # Create environment
    env = LoLEnvironment(headless=headless, use_yolo=True)

    # Create agent (random policy)
    agent = PPOAgent()

    print("Starting demo (running for 30 seconds)...")
    print("Press Ctrl+C to stop early")

    try:
        obs, info = env.reset()
        agent.reset_hidden_state()

        start_time = time.time()
        steps = 0

        while time.time() - start_time < 30.0:
            # Select random action
            action = agent.select_action(obs, deterministic=False)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)

            steps += 1

            # Print stats every 2 seconds
            if steps % 60 == 0:
                print(f"\nTime: {time.time() - start_time:.1f}s")
                print(f"HP: {info.get('player_hp', 0):.1%}")
                print(f"Detections: {info.get('detections', 0)}")
                print(f"FPS: {info.get('fps', 0):.1f}")
                print(f"APM: {info.get('apm', 0):.1f}")

            if terminated or truncated:
                obs, info = env.reset()
                agent.reset_hidden_state()

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    finally:
        env.close()

    print("\nDemo completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoL RL Agent - Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo (observation only, no input execution)
  python main.py demo

  # Run trained model in Practice Tool
  python main.py infer --model checkpoints/best_model.pt --practice-tool

  # Run trained model in headless mode (testing)
  python main.py infer --model checkpoints/best_model.pt --headless

  # Start training
  python main.py train --curriculum-stage cs_training
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Enable input execution (WARNING: will control your mouse/keyboard)"
    )

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run trained model")
    infer_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--practice-tool",
        action="store_true",
        help="Use Practice Tool environment"
    )
    infer_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no input execution)"
    )
    infer_parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )

    # Train command (redirect to train.py)
    train_parser = subparsers.add_parser("train", help="Start training (see train.py for options)")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    if args.command == "demo":
        run_demo(headless=not args.no_headless)

    elif args.command == "infer":
        run_inference(
            model_path=args.model,
            practice_tool=args.practice_tool,
            headless=args.headless,
            max_episodes=args.episodes
        )

    elif args.command == "train":
        print("For training, please use train.py directly:")
        print("  python train.py --help")
        sys.exit(0)

    else:
        print("Please specify a command: demo, infer, or train")
        print("Run with --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
