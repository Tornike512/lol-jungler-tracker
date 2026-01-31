"""
Garen AI Training Pipeline
==========================
1. Find top Garen players from leaderboards
2. Download their match history via Riot API
3. Extract game data to CSV
4. Train behavioral cloning model
"""
import requests
import json
import csv
import os
import time
import sys

# API Configuration
RIOT_API_KEY = os.environ.get('RIOT_API_KEY', 'RGAPI-e197fdd2-f5ce-442c-8c3d-bd57226a8e00')

REGIONS = {
    'EUW1': 'europe',
    'NA1': 'americas',
    'KR': 'asia',
    'EUN1': 'europe',
}

# Top Garen players (One-Trick-Ponies) - from leaderboards
# Format: (gameName, tagLine, region)
TOP_GAREN_PLAYERS = [
    # EUW - From leagueofgraphs leaderboard
    ("YTGaReN HaN live", "RTE", "EUW1"),
    ("Marteau", "EUW", "EUW1"),
    ("BOBAIL", "EUW", "EUW1"),
    ("Vildred", "EUWW", "EUW1"),
    ("No name sorry", "EUW", "EUW1"),
    ("Garenmania", "EUW", "EUW1"),
    ("Press E to Spin", "EUW", "EUW1"),
    ("xSuits", "EUW", "EUW1"),
    ("LightEternal", "Light", "EUW1"),
    ("rodai85", "EUW", "EUW1"),
    ("Aenigmanta", "EUW", "EUW1"),
    ("Garen player", "EUW", "EUW1"),
    ("Palco Granko", "split", "EUW1"),
    ("Elolesio", "SPIN", "EUW1"),
    ("solokill addict", "bleed", "EUW1"),
    # NA - From leagueofgraphs leaderboard
    ("starry eyes", "zzz", "NA1"),
    ("sounding fetish", "NA2", "NA1"),
    ("Koryx26", "NA1", "NA1"),
    ("chrisnam", "NA1", "NA1"),
    ("can u feel my", "heart", "NA1"),
    ("Taiwan Real CN", "Tibet", "NA1"),
    ("stray kid", "tag", "NA1"),
    ("Captain Facelamb", "EZER0", "NA1"),
    ("Sond", "NA1", "NA1"),
    ("AddictedToBacon", "TTV", "NA1"),
    ("Screelix", "SpinR", "NA1"),
    ("Among us pro", "NA1", "NA1"),
    ("Triton", "DMCIA", "NA1"),
    # KR - From leagueofgraphs leaderboard
    ("disabled guy", "KR1", "KR"),
    ("Kuncle Duster", "KSH", "KR"),
]

OUTPUT_CSV = "garen_training_data.csv"
MATCHES_PER_PLAYER = 10


class GarenDataCollector:
    def __init__(self):
        self.headers = {'X-Riot-Token': RIOT_API_KEY}
        self.matches_processed = 0
        self.rows_collected = 0

    def get_puuid(self, game_name, tag_line, region):
        """Get player's PUUID from Riot ID."""
        routing = REGIONS.get(region, 'europe')
        url = f"https://{routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json().get('puuid')
            elif response.status_code == 404:
                print(f"    Player not found: {game_name}#{tag_line}")
            elif response.status_code == 403:
                print(f"    API key expired or invalid!")
                return None
            else:
                print(f"    API error: {response.status_code}")
        except Exception as e:
            print(f"    Error: {e}")
        return None

    def get_match_ids(self, puuid, region, count=5):
        """Get recent match IDs for a player."""
        routing = REGIONS.get(region, 'europe')
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {'count': count, 'type': 'ranked'}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"    Error getting matches: {e}")
        return []

    def get_match_data(self, match_id, region):
        """Get full match data."""
        routing = REGIONS.get(region, 'europe')
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"    Error: {e}")
        return None

    def get_match_timeline(self, match_id, region):
        """Get match timeline with position data."""
        routing = REGIONS.get(region, 'europe')
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"    Error: {e}")
        return None

    def extract_garen_data(self, match_data, timeline_data, writer):
        """Extract Garen player data from match and write to CSV."""
        if not match_data or not timeline_data:
            return 0

        participants = match_data.get('info', {}).get('participants', [])

        # Find Garen player
        garen_player = None
        garen_id = None
        for p in participants:
            if p.get('championName', '').lower() == 'garen':
                garen_player = p
                garen_id = p.get('participantId')
                break

        if not garen_player:
            return 0

        # Extract timeline frames
        frames = timeline_data.get('info', {}).get('frames', [])
        rows_added = 0

        # Get match metadata
        game_duration = match_data.get('info', {}).get('gameDuration', 0)
        win = 1 if garen_player.get('win') else 0

        for frame in frames:
            timestamp = frame.get('timestamp', 0) / 1000  # Convert to seconds
            participant_frames = frame.get('participantFrames', {})

            garen_frame = participant_frames.get(str(garen_id))
            if not garen_frame:
                continue

            pos = garen_frame.get('position', {})
            x = pos.get('x', 0)
            y = pos.get('y', 0)

            # Skip invalid positions
            if x == 0 and y == 0:
                continue

            # Extract frame data
            level = garen_frame.get('level', 1)
            current_gold = garen_frame.get('currentGold', 0)
            total_gold = garen_frame.get('totalGold', 0)
            xp = garen_frame.get('xp', 0)
            minions_killed = garen_frame.get('minionsKilled', 0)
            jungle_minions = garen_frame.get('jungleMinionsKilled', 0)

            # Damage stats (if available)
            damage_stats = garen_frame.get('damageStats', {})
            total_damage_done = damage_stats.get('totalDamageDoneToChampions', 0)
            total_damage_taken = damage_stats.get('totalDamageTaken', 0)

            writer.writerow([
                round(timestamp, 1),
                'Garen',
                level,
                current_gold,
                total_gold,
                xp,
                x,
                y,
                minions_killed,
                jungle_minions,
                total_damage_done,
                total_damage_taken,
                win,
                game_duration
            ])
            rows_added += 1

        return rows_added

    def collect_data(self):
        """Main data collection loop."""
        print("=" * 60)
        print("GAREN AI DATA COLLECTOR")
        print("=" * 60)

        # Check API key
        print("\nChecking API key...")
        try:
            test_url = "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/test/test"
            response = requests.get(test_url, headers=self.headers, timeout=15)
            if response.status_code == 403:
                print("[!] API key is invalid or expired!")
                print("    Get a new key from: https://developer.riotgames.com/")
                return False
            print("[+] API key is valid")
        except requests.exceptions.Timeout:
            print("[!] API check timed out, continuing anyway...")
        except Exception as e:
            print(f"[!] API check error: {e}, continuing anyway...")

        # Open CSV file
        file_exists = os.path.exists(OUTPUT_CSV)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists or os.path.getsize(OUTPUT_CSV) == 0:
                writer.writerow([
                    'timestamp', 'champion', 'level', 'current_gold', 'total_gold',
                    'xp', 'x', 'y', 'minions_killed', 'jungle_minions',
                    'damage_done', 'damage_taken', 'win', 'game_duration'
                ])

            # Process each player
            for game_name, tag_line, region in TOP_GAREN_PLAYERS:
                print(f"\n[*] Processing {game_name}#{tag_line} ({region})...")

                # Get PUUID
                puuid = self.get_puuid(game_name, tag_line, region)
                if not puuid:
                    continue

                # Get match IDs
                match_ids = self.get_match_ids(puuid, region, MATCHES_PER_PLAYER)
                print(f"    Found {len(match_ids)} matches")

                for match_id in match_ids:
                    print(f"    Processing {match_id}...", end=' ')

                    # Get match data
                    match_data = self.get_match_data(match_id, region)
                    if not match_data:
                        print("failed (match data)")
                        continue

                    # Get timeline
                    timeline = self.get_match_timeline(match_id, region)
                    if not timeline:
                        print("failed (timeline)")
                        continue

                    # Extract data
                    rows = self.extract_garen_data(match_data, timeline, writer)
                    if rows > 0:
                        self.matches_processed += 1
                        self.rows_collected += rows
                        print(f"+{rows} rows")
                    else:
                        print("no Garen data")

                    # Rate limiting
                    time.sleep(1.2)

        print("\n" + "=" * 60)
        print(f"COLLECTION COMPLETE")
        print(f"  Matches processed: {self.matches_processed}")
        print(f"  Total rows: {self.rows_collected}")
        print(f"  Output file: {OUTPUT_CSV}")
        print("=" * 60)

        return True


def main():
    collector = GarenDataCollector()
    collector.collect_data()


if __name__ == "__main__":
    main()
