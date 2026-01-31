"""
Garen AI Interactive Tester
============================
Test the AI with custom game states.
"""
from garen_predict import GarenPredictor


def main():
    print("=" * 60)
    print("GAREN AI INTERACTIVE TESTER")
    print("=" * 60)

    predictor = GarenPredictor()

    print("\nEnter game state values (or press Enter for defaults)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            print("-" * 40)

            # Get inputs
            x = input("Current X position [7500]: ").strip()
            if x.lower() == 'quit':
                break
            x = int(x) if x else 7500

            y = input("Current Y position [7500]: ").strip()
            if y.lower() == 'quit':
                break
            y = int(y) if y else 7500

            level = input("Level [1-18, default 6]: ").strip()
            level = int(level) if level else 6

            cs = input("Minions killed (CS) [0]: ").strip()
            cs = int(cs) if cs else 0

            game_time = input("Game time in minutes [10]: ").strip()
            game_time = (int(game_time) * 60) if game_time else 600

            gold = input("Total gold [3000]: ").strip()
            gold = int(gold) if gold else 3000

            winning = input("Winning? (y/n) [y]: ").strip().lower()
            win = 0 if winning == 'n' else 1

            # Create game state
            state = {
                'x': x,
                'y': y,
                'level': level,
                'current_gold': gold // 2,
                'total_gold': gold,
                'xp': level * 500,
                'minions_killed': cs,
                'jungle_minions': 0,
                'damage_done': level * 1000,
                'damage_taken': level * 800,
                'game_time': game_time,
                'game_duration': 1800,
                'win': win
            }

            # Predict
            prediction = predictor.predict_position(state)

            print(f"\n>>> PREDICTION <<<")
            print(f"Current position: ({x}, {y})")
            print(f"Suggested position: ({prediction['x']}, {prediction['y']})")
            print(f"Location: {prediction['description']}")

            # Direction
            dx = prediction['x'] - x
            dy = prediction['y'] - y
            distance = (dx**2 + dy**2) ** 0.5

            print(f"Distance to move: {distance:.0f} units")

            if distance > 500:
                if dx > 0:
                    print("  -> Move RIGHT")
                elif dx < 0:
                    print("  -> Move LEFT")
                if dy > 0:
                    print("  -> Move UP (towards top lane)")
                elif dy < 0:
                    print("  -> Move DOWN (towards bot lane)")
            else:
                print("  -> Stay at current position")

            print()

        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
