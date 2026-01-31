import requests
import os
import time
import urllib3
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# Get from developer.riotgames.com
RIOT_API_KEY = "RGAPI-e197fdd2-f5ce-442c-8c3d-bd57226a8e00"
REGION = "euw1"  # Change based on the player's region (jp1, kr, na1, euw1)
ROUTING = "europe"  # (asia, americas, europe)
# ---------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GarenProScraper:
    def __init__(self):
        self.lockfile_path = r"C:\Riot Games\League of Legends\lockfile"
        self.auth = None
        self.lcu_url = None

    def get_lcu_auth(self):
        if not os.path.isfile(self.lockfile_path):
            return False
        with open(self.lockfile_path, 'r') as f:
            _, _, port, password, _ = f.read().split(':')
            self.auth = requests.auth.HTTPBasicAuth('riot', password)
            self.lcu_url = f"https://127.0.0.1:{port}"
        return True

    def get_puuid_from_riot(self, game_name, tag_line):
        """Uses the Web API to find anyone's PUUID instantly."""
        url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        params = {"api_key": RIOT_API_KEY}
        res = requests.get(url, params=params)
        if res.status_code == 200:
            return res.json()['puuid']
        print(f"  [!] Failed to find {game_name}#{tag_line} on Riot servers.")
        return None

    def get_match_id(self, puuid):
        """Finds the latest Match ID for a PUUID."""
        url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"api_key": RIOT_API_KEY, "count": 1}
        res = requests.get(url, params=params)
        if res.status_code == 200 and len(res.json()) > 0:
            # Match IDs look like "JP1_12345", LCU just wants the "12345"
            full_id = res.json()[0]
            return full_id.split('_')[1]
        return None

    def trigger_lcu_download(self, match_id):
        """Tells your local client to download the specific file."""
        url = f"{self.lcu_url}/lol-replays/v1/rofls/{match_id}/download"
        res = requests.post(url, auth=self.auth, verify=False)
        return res.status_code == 204

    def run(self):
        if not self.get_lcu_auth():
            print("Please open League of Legends first.")
            return

        # Replace these with the names you scraped or found
        # Use these in your script's 'targets' list
        targets = [
            ("nero2108", "EUW"),
            ("Rezist", "EUW"),
            ("rodai85", "EUW"),
            ("Aenigmanta", "EUW"),
            ("Garen player", "EUW")
        ]

        for name, tag in targets:
            print(f"Processing {name}#{tag}...")
            puuid = self.get_puuid_from_riot(name, tag)
            if puuid:
                match_id = self.get_match_id(puuid)
                if match_id:
                    if self.trigger_lcu_download(match_id):
                        print(f"  [+] Download started for {match_id}!")
                    else:
                        print(
                            f"  [!] Client refused download (Already have it?)")
            time.sleep(1)  # Respect API rate limits


if __name__ == "__main__":
    GarenProScraper().run()
