"""
Get match data from Riot API using match ID from replay filename.
The filename format is: REGION-MATCHID.rofl (e.g., EUW1-7709857473.rofl)
"""
import requests
import json
import csv
import sys
import os
import re

# Get API key from rofl_scraper.py or environment
RIOT_API_KEY = os.environ.get('RIOT_API_KEY', 'RGAPI-e197fdd2-f5ce-442c-8c3d-bd57226a8e00')

REGION_ROUTING = {
    'EUW1': 'europe',
    'EUN1': 'europe',
    'NA1': 'americas',
    'BR1': 'americas',
    'LA1': 'americas',
    'LA2': 'americas',
    'KR': 'asia',
    'JP1': 'asia',
    'OC1': 'asia',
    'TR1': 'europe',
    'RU': 'europe',
}


def get_match_data(match_id, routing='europe'):
    """Fetch match data from Riot API."""
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {'X-Riot-Token': RIOT_API_KEY}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        print("[!] API key expired or invalid. Get a new one from developer.riotgames.com")
        return None
    elif response.status_code == 404:
        print(f"[!] Match not found: {match_id}")
        return None
    else:
        print(f"[!] API error: {response.status_code} - {response.text}")
        return None


def get_match_timeline(match_id, routing='europe'):
    """Fetch match timeline with position data from Riot API."""
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {'X-Riot-Token': RIOT_API_KEY}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"[!] Timeline API error: {response.status_code}")
        return None


def extract_from_rofl_filename(filename):
    """Extract region and match ID from rofl filename."""
    basename = os.path.basename(filename)
    match = re.match(r'([A-Z0-9]+)-(\d+)\.rofl', basename)
    if match:
        region = match.group(1)
        game_id = match.group(2)
        return region, f"{region}_{game_id}"
    return None, None


def process_match(rofl_path):
    """Process a single replay file and extract Katarina data."""
    print(f"--- MATCH DATA EXTRACTOR ---")
    print(f"File: {rofl_path}\n")

    region, match_id = extract_from_rofl_filename(rofl_path)
    if not match_id:
        print("[!] Could not parse match ID from filename")
        return

    routing = REGION_ROUTING.get(region, 'europe')
    print(f"Region: {region}")
    print(f"Match ID: {match_id}")
    print(f"Routing: {routing}\n")

    # Get match data
    match_data = get_match_data(match_id, routing)
    if not match_data:
        return

    # Find Katarina
    participants = match_data.get('info', {}).get('participants', [])
    katarina = None
    katarina_id = None

    print("Players:")
    print("-" * 60)
    for p in participants:
        champ = p.get('championName', 'Unknown')
        name = p.get('summonerName', p.get('riotIdGameName', 'Unknown'))
        team = 'Blue' if p.get('teamId') == 100 else 'Red'
        kills = p.get('kills', 0)
        deaths = p.get('deaths', 0)
        assists = p.get('assists', 0)
        win = 'Win' if p.get('win') else 'Loss'

        marker = ""
        if champ.lower() == 'katarina':
            katarina = p
            katarina_id = p.get('participantId')
            marker = " <-- TARGET"

        print(f"  [{team:4}] {champ:15} {name[:18]:18} KDA: {kills}/{deaths}/{assists} ({win}){marker}")

    print("-" * 60)

    if not katarina:
        print("\n[!] Katarina not found in this match.")
        return

    print(f"\n[+] Found Katarina!")
    print(f"    Participant ID: {katarina_id}")
    print(f"    Result: {'Win' if katarina.get('win') else 'Loss'}")
    print(f"    KDA: {katarina.get('kills')}/{katarina.get('deaths')}/{katarina.get('assists')}")
    print(f"    Gold Earned: {katarina.get('goldEarned'):,}")
    print(f"    Level: {katarina.get('champLevel')}")

    # Get timeline for position data
    print("\n[*] Fetching match timeline for position data...")
    timeline = get_match_timeline(match_id, routing)

    if timeline:
        frames = timeline.get('info', {}).get('frames', [])
        print(f"    Found {len(frames)} timeline frames (1 per minute)")

        # Extract Katarina position data
        output_file = 'katarina_training_data.csv'
        rows_added = 0

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['timestamp', 'champion', 'hp', 'max_hp',
                                'x', 'y', 'z', 'is_dead', 'level', 'current_gold'])

            for frame in frames:
                timestamp = frame.get('timestamp', 0) / 1000  # Convert to seconds
                participant_frames = frame.get('participantFrames', {})

                kat_frame = participant_frames.get(str(katarina_id))
                if kat_frame:
                    pos = kat_frame.get('position', {})
                    x = pos.get('x', 0)
                    y = pos.get('y', 0)

                    # Skip spawn/invalid positions
                    if x == 0 and y == 0:
                        continue

                    hp = kat_frame.get('currentGold', 0)  # Timeline doesn't have HP
                    level = kat_frame.get('level', 1)
                    gold = kat_frame.get('currentGold', 0)
                    total_gold = kat_frame.get('totalGold', 0)

                    # Use total gold and level as proxy for state
                    writer.writerow([
                        round(timestamp, 1),
                        'Katarina',
                        0,  # HP not available in timeline
                        0,  # Max HP not available
                        x,
                        y,
                        0,  # Z not available
                        0,  # isDead - would need event parsing
                        level,
                        gold
                    ])
                    rows_added += 1

        print(f"\n[+] Added {rows_added} position samples to {output_file}")
        print(f"    Note: Timeline provides 1 sample per minute (not 10Hz)")
    else:
        print("    Could not fetch timeline data")

    # Save full match data
    with open('match_data.json', 'w') as f:
        json.dump(match_data, f, indent=2)
    print(f"\n[+] Full match data saved to match_data.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_match_data.py <replay.rofl>")
        print("\nExtracts match data using Riot API from replay filename.")
        return

    process_match(sys.argv[1])


if __name__ == "__main__":
    main()
