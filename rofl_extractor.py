import json
import csv
import os
import sys

OUTPUT_FILE = "katarina_training_data.csv"


def extract_katarina_data(json_file):
    """
    Extract Katarina position data from ROFL-parsed JSON file.
    The JSON contains champion positions at 1-second intervals.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find Katarina's player index from metadata
    katarina_player = None
    players = data.get('players', [])

    for player in players:
        champion = player.get('champion', '') or player.get('championName', '')
        if champion.lower() == 'katarina':
            katarina_player = player
            break

    if not katarina_player:
        print("[!] Katarina not found in this replay.")
        print("Champions in this game:")
        for p in players:
            champ = p.get('champion', '') or p.get('championName', '')
            name = p.get('summonerName', '') or p.get('name', '')
            print(f"  - {champ} ({name})")
        return 0

    # Get Katarina's participant ID or index
    participant_id = katarina_player.get('participantId', katarina_player.get('id', 0))
    summoner_name = katarina_player.get('summonerName', '') or katarina_player.get('name', 'Unknown')

    print(f"[+] Found Katarina played by: {summoner_name}")

    # Extract position data
    positions = data.get('positions', data.get('playerPositions', {}))
    rows_saved = 0

    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(['timestamp', 'champion', 'hp', 'max_hp',
                            'x', 'y', 'z', 'is_dead', 'level', 'current_gold'])

        # Handle different JSON structures from ROFL parser
        if isinstance(positions, dict):
            # Structure: {timestamp: {playerId: {x, y, ...}}}
            for timestamp, player_positions in positions.items():
                if str(participant_id) in player_positions:
                    pos = player_positions[str(participant_id)]
                    x = pos.get('x', 0)
                    y = pos.get('y', 0)
                    z = pos.get('z', 0)
                    hp = pos.get('hp', pos.get('currentHealth', 100))
                    max_hp = pos.get('maxHp', pos.get('maxHealth', 100))
                    level = pos.get('level', 1)
                    gold = pos.get('gold', pos.get('currentGold', 0))
                    is_dead = pos.get('isDead', 0)

                    # Data validation: skip invalid states
                    if hp == 0 or (x == 0 and z == 0):
                        continue

                    writer.writerow([
                        float(timestamp),
                        "Katarina",
                        round(hp, 2),
                        round(max_hp, 2),
                        round(x, 2),
                        round(y, 2),
                        round(z, 2),
                        int(is_dead),
                        level,
                        round(gold, 2)
                    ])
                    rows_saved += 1

        elif isinstance(positions, list):
            # Structure: [{timestamp, playerId, x, y, ...}, ...]
            for pos in positions:
                pid = pos.get('participantId', pos.get('playerId', pos.get('id', -1)))
                if pid == participant_id:
                    x = pos.get('x', 0)
                    y = pos.get('y', 0)
                    z = pos.get('z', 0)
                    timestamp = pos.get('timestamp', pos.get('time', 0))
                    hp = pos.get('hp', pos.get('currentHealth', 100))
                    max_hp = pos.get('maxHp', pos.get('maxHealth', 100))
                    level = pos.get('level', 1)
                    gold = pos.get('gold', pos.get('currentGold', 0))
                    is_dead = pos.get('isDead', 0)

                    # Data validation: skip invalid states
                    if hp == 0 or (x == 0 and z == 0):
                        continue

                    writer.writerow([
                        float(timestamp),
                        "Katarina",
                        round(hp, 2),
                        round(max_hp, 2),
                        round(x, 2),
                        round(y, 2),
                        round(z, 2),
                        int(is_dead),
                        level,
                        round(gold, 2)
                    ])
                    rows_saved += 1

    return rows_saved


def main():
    if len(sys.argv) < 2:
        print("Usage: python rofl_extractor.py <replay_data.json>")
        print("\nFirst, parse your .rofl file with ROFL.exe:")
        print("  ROFL.exe file -r replay.rofl -o replay_data.json")
        print("\nThen extract Katarina data:")
        print("  python rofl_extractor.py replay_data.json")
        return

    json_file = sys.argv[1]

    if not os.path.exists(json_file):
        print(f"[!] File not found: {json_file}")
        return

    print(f"--- 🗡️ KATARINA ROFL EXTRACTOR ---")
    print(f"Processing: {json_file}")

    rows = extract_katarina_data(json_file)

    if rows > 0:
        print(f"\n[+] Extracted {rows} position samples to {OUTPUT_FILE}")
    else:
        print(f"\n[!] No valid Katarina data found in replay.")


if __name__ == "__main__":
    main()
