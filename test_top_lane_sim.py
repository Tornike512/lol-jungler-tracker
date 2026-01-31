"""
Test script for the LoL Top Lane Simulator.

Examples:
    python test_top_lane_sim.py               # Random agent with rendering
    python test_top_lane_sim.py --headless    # Random agent, no rendering
    python test_top_lane_sim.py --heuristic   # Simple heuristic agent
    python test_top_lane_sim.py --flat        # Test flat action space
"""

import sys
import numpy as np
from src.lol_top_lane_sim import LoLTopLaneEnv, LoLTopLaneFlatEnv

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame_ce as pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False


def run_random_agent(render: bool = True, max_steps: int = 2000):
    """Run the environment with random actions."""
    print("=" * 60)
    print("Random Agent Test")
    print("=" * 60)

    render_mode = "human" if render and PYGAME_AVAILABLE else None
    env = LoLTopLaneEnv(render_mode=render_mode, simulation_speed=2.0)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial position: {info['garen_position']}")

    total_reward = 0
    done = False
    step = 0

    try:
        while not done and step < max_steps:
            # Random action
            action = {
                "move": env.action_space["move"].sample(),
                "attack": env.action_space["attack"].sample()
            }

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render_mode:
                env.render()

            if step % 200 == 0:
                print(f"Step {step:4d} | Reward: {reward:+.3f} | "
                      f"Total: {total_reward:+.2f} | CS: {info['last_hits']} | "
                      f"Gold: {info['gold']}")

            step += 1

            # Handle Pygame window close
            if render_mode and PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print(f"\n{'=' * 60}")
    print(f"Episode Summary:")
    print(f"  Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Last Hits (CS): {info['last_hits']}")
    print(f"  Missed Last Hits: {info['missed_last_hits']}")
    print(f"  Minions Lost: {info['minions_missed']}")
    print(f"  Gold Earned: {info['gold']}")
    print(f"  CS/min: {info['cs_per_minute']:.1f}")

    env.close()


def run_heuristic_agent(render: bool = True, max_steps: int = 3000):
    """
    Run a simple heuristic agent that:
    1. Walks toward the lane
    2. Approaches low-HP minions
    3. Attacks when in range and minion HP is low enough
    """
    print("=" * 60)
    print("Heuristic Agent Test")
    print("=" * 60)

    render_mode = "human" if render and PYGAME_AVAILABLE else None
    env = LoLTopLaneEnv(render_mode=render_mode, simulation_speed=1.0)

    obs, info = env.reset()

    # Lane target (middle of top lane)
    lane_target = (300, 300)

    total_reward = 0
    done = False
    step = 0

    try:
        while not done and step < max_steps:
            # Parse observation
            garen_x = obs[0] * 1000  # De-normalize
            garen_y = obs[1] * 1000
            attack_ready = obs[2] > 0.5

            # Get minion info (3 closest)
            minions = []
            for i in range(3):
                base_idx = 3 + i * 3
                mx, my, hp = obs[base_idx], obs[base_idx + 1], obs[base_idx + 2]
                if hp > 0:  # Minion exists
                    minions.append({
                        'x': mx * 1000,
                        'y': my * 1000,
                        'hp': hp,
                        'idx': i
                    })

            # Decision making
            attack_idx = 6  # No attack by default

            if minions:
                # Find the lowest HP minion in attack range (175 units)
                best_target = None
                best_hp = 1.0

                for m in minions:
                    dist = np.sqrt((m['x'] - garen_x)**2 + (m['y'] - garen_y)**2)
                    # Attack if in range and low HP (prefer lower HP targets)
                    if dist <= 175 and m['hp'] < best_hp:
                        # Only attack if HP is low enough (simulating last-hit timing)
                        # Assume we can kill if HP < 15%
                        if m['hp'] < 0.15 and attack_ready:
                            best_target = m
                            best_hp = m['hp']

                if best_target:
                    attack_idx = best_target['idx']

                # Move toward lowest HP minion
                target_minion = min(minions, key=lambda m: m['hp'])
                dx = target_minion['x'] - garen_x
                dy = target_minion['y'] - garen_y
                dist_to_minion = np.sqrt(dx**2 + dy**2)

                if dist_to_minion > 150:  # Move closer
                    angle = np.arctan2(dy, dx)
                    magnitude = 0.8
                else:
                    # In range, stop or circle slightly
                    angle = np.arctan2(dy, dx)
                    magnitude = 0.1
            else:
                # No minions visible, walk toward lane
                dx = lane_target[0] - garen_x
                dy = lane_target[1] - garen_y
                angle = np.arctan2(dy, dx)
                magnitude = 1.0

            # Build action
            action = {
                "move": np.array([angle, magnitude], dtype=np.float32),
                "attack": attack_idx
            }

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render_mode:
                env.render()

            if step % 200 == 0:
                print(f"Step {step:4d} | Reward: {reward:+.3f} | "
                      f"Total: {total_reward:+.2f} | CS: {info['last_hits']} | "
                      f"Gold: {info['gold']} | Minions: {len(minions)}")

            step += 1

            # Handle Pygame events
            if render_mode and PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print(f"\n{'=' * 60}")
    print(f"Heuristic Agent Summary:")
    print(f"  Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Last Hits (CS): {info['last_hits']}")
    print(f"  Missed Last Hits: {info['missed_last_hits']}")
    print(f"  Minions Lost: {info['minions_missed']}")
    print(f"  Gold Earned: {info['gold']}")
    print(f"  CS/min: {info['cs_per_minute']:.1f}")

    env.close()


def test_flat_env():
    """Test the flat action space environment."""
    print("=" * 60)
    print("Flat Action Space Test")
    print("=" * 60)

    env = LoLTopLaneFlatEnv(render_mode=None)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"\nAfter 500 steps:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  CS: {info['last_hits']}")
    print(f"  Gold: {info['gold']}")

    env.close()
    print("\nFlat env works correctly!")


def main():
    args = sys.argv[1:]

    if "--flat" in args:
        test_flat_env()
    elif "--heuristic" in args:
        render = "--headless" not in args
        run_heuristic_agent(render=render)
    else:
        render = "--headless" not in args
        run_random_agent(render=render)


if __name__ == "__main__":
    main()
