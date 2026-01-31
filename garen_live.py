"""
Garen AI Live Game Assistant
=============================
Connects to a live League game and provides real-time positioning advice.
Run this while playing Garen in Practice Tool or a real game.
"""
import requests
import urllib3
import time
import sys

from garen_predict import GarenPredictor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"


def get_game_data():
    """Fetch live game data from Riot API."""
    try:
        response = requests.get(API_URL, verify=False, timeout=1)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def find_garen(game_data):
    """Find Garen in the player list."""
    for player in game_data.get('allPlayers', []):
        if player.get('championName', '').lower() == 'garen':
            return player
    return None


def extract_game_state(game_data, garen_data):
    """Extract game state for the AI model."""
    stats = garen_data.get('championStats', {})
    pos = garen_data.get('position', {})
    scores = garen_data.get('scores', {})

    game_time = game_data.get('gameData', {}).get('gameTime', 0)

    # Determine if winning (compare team gold/kills)
    my_team = garen_data.get('team', 'ORDER')
    team_kills = 0
    enemy_kills = 0
    for p in game_data.get('allPlayers', []):
        if p.get('team') == my_team:
            team_kills += p.get('scores', {}).get('kills', 0)
        else:
            enemy_kills += p.get('scores', {}).get('kills', 0)

    win = 1 if team_kills > enemy_kills else 0

    return {
        'x': pos.get('x', 7500),
        'y': pos.get('y', 7500),
        'level': garen_data.get('level', 1),
        'current_gold': scores.get('currentGold', 0),
        'total_gold': scores.get('currentGold', 0) * 2,  # Estimate
        'xp': garen_data.get('level', 1) * 500,  # Estimate
        'minions_killed': scores.get('creepScore', 0),
        'jungle_minions': 0,
        'damage_done': stats.get('physicalDamageDealtToChampions', 0),
        'damage_taken': stats.get('physicalDamageTaken', 0),
        'game_time': game_time,
        'game_duration': 1800,  # Assume 30 min game
        'win': win
    }


def main():
    print("=" * 60)
    print("GAREN AI LIVE ASSISTANT")
    print("=" * 60)
    print("\nStart a game as Garen (Practice Tool recommended)")
    print("Press Ctrl+C to stop\n")

    predictor = GarenPredictor()

    print("\nWaiting for game...")

    last_prediction = None

    try:
        while True:
            game_data = get_game_data()

            if not game_data:
                print("No game detected. Waiting...", end='\r')
                time.sleep(2)
                continue

            garen = find_garen(game_data)

            if not garen:
                print("Garen not found in game. Play as Garen!", end='\r')
                time.sleep(1)
                continue

            # Extract state and predict
            state = extract_game_state(game_data, garen)
            prediction = predictor.predict_position(state)

            # Display
            game_time = state['game_time']
            minutes = int(game_time // 60)
            seconds = int(game_time % 60)

            status = "WINNING" if state['win'] else "LOSING"

            print(f"\n[{minutes:02d}:{seconds:02d}] Level {state['level']} | CS: {state['minions_killed']} | {status}")
            print(f"  Current:   ({state['x']:.0f}, {state['y']:.0f})")
            print(f"  Suggested: ({prediction['x']}, {prediction['y']}) - {prediction['description']}")

            # Give directional advice
            dx = prediction['x'] - state['x']
            dy = prediction['y'] - state['y']

            if abs(dx) > 500 or abs(dy) > 500:
                directions = []
                if dx > 500:
                    directions.append("RIGHT (towards red side)")
                elif dx < -500:
                    directions.append("LEFT (towards blue side)")
                if dy > 500:
                    directions.append("UP (towards top)")
                elif dy < -500:
                    directions.append("DOWN (towards bot)")

                if directions:
                    print(f"  >> Move: {', '.join(directions)}")
            else:
                print(f"  >> Good position! Stay here.")

            time.sleep(1)  # Update every second

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
