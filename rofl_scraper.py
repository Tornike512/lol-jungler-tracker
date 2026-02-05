import requests
import base64
import os
import time
import shutil
import glob
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RIOT_API_KEY = os.environ.get("RIOT_API_KEY", "")
RIOT_HEADERS = {"X-Riot-Token": RIOT_API_KEY}

# Your account
PLAYER_NAME = "FAIRY PEPE"
PLAYER_TAG = "EUW"
CHAMPION_NAME = "Katarina"
CHAMPION_ID = 55
MAX_MATCHES = 20

LOCKFILE_PATH = r"C:\Riot Games\League of Legends\lockfile"
ROFLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rofls")

# --- Riot API ---

def riot_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=RIOT_HEADERS, params=params, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                print(f"  Timeout, retrying ({attempt+1}/{retries})...", flush=True)
                time.sleep(3)
            else:
                print(f"  Request failed after {retries} attempts")
    return None

def get_puuid(game_name, tag_line):
    url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    r = riot_get(url)
    if r and r.status_code == 200:
        return r.json()["puuid"]
    if r:
        print(f"  Could not find account {game_name}#{tag_line}: {r.status_code} {r.text}")
    return None

def get_match_ids(puuid, count=20):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"count": count}
    r = riot_get(url, params=params)
    if r and r.status_code == 200:
        return r.json()
    if r:
        print(f"  Could not fetch matches: {r.status_code} {r.text}")
    return []

def get_match_detail(match_id):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = riot_get(url)
    if r and r.status_code == 200:
        return r.json()
    return None

def extract_game_id(match_id_str):
    parts = match_id_str.split("_")
    if len(parts) == 2:
        return int(parts[1])
    return None

def fetch_katarina_game_ids():
    print(f"\nLooking up {PLAYER_NAME}#{PLAYER_TAG}...")
    puuid = get_puuid(PLAYER_NAME, PLAYER_TAG)
    if not puuid:
        return []
    time.sleep(1)

    print(f"  Fetching recent matches...")
    match_ids = get_match_ids(puuid, count=MAX_MATCHES * 3)
    time.sleep(1)

    game_ids = []
    for mid in match_ids:
        if len(game_ids) >= MAX_MATCHES:
            break
        detail = get_match_detail(mid)
        time.sleep(1.2)
        if not detail:
            continue

        participants = detail.get("info", {}).get("participants", [])
        played_katarina = any(
            p["puuid"] == puuid and p["championId"] == CHAMPION_ID
            for p in participants
        )
        if played_katarina:
            gid = extract_game_id(mid)
            if gid:
                print(f"  Found {CHAMPION_NAME} game: {mid} (gameId: {gid})")
                game_ids.append(gid)

    print(f"  Found {len(game_ids)} {CHAMPION_NAME} games")
    return game_ids

# --- LCU ---

def get_lcu_credentials():
    if not os.path.isfile(LOCKFILE_PATH):
        print("Error: Lockfile not found. Is the League client open?")
        return None, None
    with open(LOCKFILE_PATH, 'r') as f:
        data = f.read().split(':')
        return data[2], data[3]

def lcu_request(method, endpoint, json_body=None):
    port, password = get_lcu_credentials()
    if not port:
        return None
    auth = base64.b64encode(f"riot:{password}".encode()).decode()
    url = f"https://127.0.0.1:{port}{endpoint}"
    try:
        r = requests.request(method, url, headers={"Authorization": f"Basic {auth}"}, json=json_body, verify=False)
        return r
    except:
        return None

def get_replay_dir():
    r = lcu_request("GET", "/lol-replays/v1/configuration")
    if r and r.status_code == 200:
        data = r.json()
        path = data.get("replaysPath") or data.get("replayPath") or data.get("replays-path")
        if path:
            return path
    default = os.path.join(os.path.expanduser("~"), "Documents", "League of Legends", "Replays")
    if os.path.isdir(default):
        return default
    os.makedirs(default, exist_ok=True)
    return default

def get_replay_state(match_id):
    r = lcu_request("GET", f"/lol-replays/v1/metadata/{match_id}")
    if r and r.status_code == 200:
        data = r.json()
        progress = data.get("downloadProgress", 0)
        state = data.get("state", "unknown")
        if state != "watch" and 0 < progress < 100:
            return f"downloading ({progress}%)"
        return state
    if r and r.status_code == 404:
        return "not_found"
    return "unknown"

def download_replay(match_id):
    r = lcu_request("POST", f"/lol-replays/v1/rofls/{match_id}/download/graceful", {"gameId": match_id})
    return r and r.status_code in [200, 204]

def wait_for_download(match_id, timeout=180):
    last_state = None
    for _ in range(timeout // 3):
        state = get_replay_state(match_id)
        if state != last_state:
            print(f"[{state}]", end=" ", flush=True)
            last_state = state
        if state == "watch":
            return True
        if state in ["lost", "incompatible"]:
            return False
        time.sleep(3)
    return False

def move_rofl(match_id, replay_dir):
    pattern = os.path.join(replay_dir, f"*{match_id}*.rofl")
    matches = glob.glob(pattern)
    if matches:
        src = matches[0]
        dst = os.path.join(ROFLS_DIR, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
        return dst
    return None

# --- Main ---

if __name__ == "__main__":
    if not RIOT_API_KEY:
        print("Error: Set RIOT_API_KEY environment variable.")
        print("  Windows: set RIOT_API_KEY=RGAPI-xxxx-xxxx-xxxx")
        exit(1)

    print(f"=== Fetching {PLAYER_NAME}#{PLAYER_TAG} {CHAMPION_NAME} games ===")
    game_ids = fetch_katarina_game_ids()

    if not game_ids:
        print(f"\nNo {CHAMPION_NAME} games found. Check API key or play some games!")
        exit(1)

    print(f"\n=== Found {len(game_ids)} {CHAMPION_NAME} games ===")

    os.makedirs(ROFLS_DIR, exist_ok=True)
    replay_dir = get_replay_dir()
    print(f"League replay dir: {replay_dir}")
    print(f"Saving to: {ROFLS_DIR}")

    print(f"\nDownloading replays...")
    downloaded = 0
    skipped_incompatible = 0

    for m_id in game_ids:
        # Already in rofls/?
        existing = glob.glob(os.path.join(ROFLS_DIR, f"*{m_id}*.rofl"))
        if existing:
            print(f"SKIPPING: {m_id} (already in rofls/)")
            downloaded += 1
            continue

        # Check state before attempting download
        state = get_replay_state(m_id)
        if state in ["incompatible", "not_found"]:
            print(f"SKIPPING: {m_id} (incompatible - old patch)")
            skipped_incompatible += 1
            continue

        # Already downloaded in League's replay dir but not moved?
        if state == "watch":
            moved = move_rofl(m_id, replay_dir)
            if moved:
                print(f"MOVED: {m_id} -> {os.path.basename(moved)}")
                downloaded += 1
                continue

        # Download
        if download_replay(m_id):
            print(f"FETCHING: {m_id} ...", end=" ", flush=True)
            if wait_for_download(m_id):
                moved = move_rofl(m_id, replay_dir)
                if moved:
                    print(f"OK -> {os.path.basename(moved)}")
                    downloaded += 1
                else:
                    print("DONE (file not found in replay dir, check manually)")
            else:
                print("UNAVAILABLE")
        else:
            print(f"SKIPPING: {m_id} (request failed)")

    print(f"\n=== Results ===")
    print(f"Downloaded/found: {downloaded}")
    print(f"Incompatible (old patch): {skipped_incompatible}")
    print(f"Replays saved to: {ROFLS_DIR}")
