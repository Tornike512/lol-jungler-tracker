import requests
import os
import time
import urllib3
import urllib.parse

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- USER CONFIG ---
RIOT_API_KEY = "RGAPI-e197fdd2-f5ce-442c-8c3d-bd57226a8e00"
ROUTING = "europe"
LOCKFILE_PATH = r"C:\Riot Games\League of Legends\lockfile"
# -------------------


class GarenCollector:
    def __init__(self):
        self.auth = None
        self.lcu_url = None
        self.replay_path = None

    def connect_lcu(self):
        if not os.path.isfile(LOCKFILE_PATH):
            print("ERROR: League Client not found. Is it open?")
            return False

        with open(LOCKFILE_PATH, 'r') as f:
            parts = f.read().split(':')
            self.auth = requests.auth.HTTPBasicAuth('riot', parts[3])
            self.lcu_url = f"https://127.0.0.1:{parts[2]}"

        try:
            # Try to get the path from the client
            res = requests.get(
                f"{self.lcu_url}/lol-replays/v1/configuration", auth=self.auth, verify=False)
            self.replay_path = res.json().get('replays-path')
        except:
            pass

        # FALLBACK: If client returns None, use the standard Windows path
        if not self.replay_path:
            self.replay_path = os.path.join(os.path.expanduser(
                "~"), "Documents", "League of Legends", "Replays")

        print(f"Connected! Target Folder: {self.replay_path}")

        # Ensure the folder actually exists
        if not os.path.exists(self.replay_path):
            os.makedirs(self.replay_path)
            print(f"Created folder: {self.replay_path}")

        return True

    def get_match_for_player(self, name, tag):
        print(f"--- Processing {name}#{tag} ---")
        enc_name = urllib.parse.quote(name)

        # 1. Get PUUID
        acc_url = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{enc_name}/{tag}?api_key={RIOT_API_KEY}"
        acc_res = requests.get(acc_url)
        if acc_res.status_code != 200:
            print(f"  [!] Failed to find account ({acc_res.status_code})")
            return None

        puuid = acc_res.json()['puuid']

        # 2. Get Match ID (Checking last 5 games to find a recent one)
        match_url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=5&api_key={RIOT_API_KEY}"
        m_res = requests.get(match_url)
        if m_res.status_code == 200 and len(m_res.json()) > 0:
            # Return only the numeric part of the ID (e.g., from EUW1_12345 to 12345)
            full_id = m_res.json()[0]
            print(f"  [*] Found Match: {full_id}")
            return full_id.split('_')[1]
        return None

    def download(self, match_id):
        # We check if the file already exists first
        for file in os.listdir(self.replay_path):
            if match_id in file:
                print(f"  [!] File {match_id} already in folder. Skipping.")
                return False

        url = f"{self.lcu_url}/lol-replays/v1/rofls/{match_id}/download"
        res = requests.post(url, auth=self.auth, verify=False)

        if res.status_code == 204:
            print(f"  [+] Command sent! Check folder in 30 seconds.")
            return True
        else:
            print(f"  [!] Client rejected download. Status: {res.status_code}")
            return False

    def run(self):
        if self.connect_lcu():
            # Using the names we know are active on EUW
            targets = [("nero2108", "EUW"),
                       ("Rezist", "EUW"), ("rodai85", "EUW")]
            for n, t in targets:
                m_id = self.get_match_for_player(n, t)
                if m_id:
                    self.download(m_id)
                time.sleep(2)


if __name__ == "__main__":
    GarenCollector().run()
