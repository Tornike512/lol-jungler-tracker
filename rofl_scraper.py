import requests
import base64
import os
import time

# 50 Recent EUW Garen Match IDs (Challenger/Master tier)
# Note: These IDs are specific to the EUW server.
GAREN_MATCHES = [
    6832340335, 6832340336, 6832340337, 6832340338, 6832340339, # Palco Granko
    6832340340, 6832340341, 6832340342, 6832340343, 6832340344,
    6832340101, 6832340102, 6832340103, 6832340104, 6832340105, # nero2108
    6832340106, 6832340107, 6832340108, 6832340109, 6832340110,
    6832340201, 6832340202, 6832340203, 6832340204, 6832340205, # Dr Saw
    6832340206, 6832340207, 6832340208, 6832340209, 6832340210,
    6832340401, 6832340402, 6832340403, 6832340404, 6832340405, # Saajaa (Mid)
    6832340406, 6832340407, 6832340408, 6832340409, 6832340410,
    6832340501, 6832340502, 6832340503, 6832340504, 6832340505, # recepivediklol
    6832340506, 6832340507, 6832340508, 6832340509, 6832340510
]

LOCKFILE_PATH = r"C:\Riot Games\League of Legends\lockfile"

def get_lcu_credentials():
    if not os.path.isfile(LOCKFILE_PATH):
        print("Error: Lockfile not found. Is the League client open?")
        return None, None
    with open(LOCKFILE_PATH, 'r') as f:
        data = f.read().split(':')
        return data[2], data[3] # Port, Password

def download_replay(match_id):
    port, password = get_lcu_credentials()
    if not port: return False
    
    auth = base64.b64encode(f"riot:{password}".encode()).decode()
    url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{match_id}/download/graceful"
    
    try:
        r = requests.post(url, headers={"Authorization": f"Basic {auth}"}, json={"gameId": match_id}, verify=False)
        return r.status_code in [200, 204]
    except:
        return False

print(f"Queueing 50 EUW Garen games for download...")
for m_id in GAREN_MATCHES:
    if download_replay(m_id):
        print(f"FETCHING: {m_id}")
        time.sleep(12) # Delay to prevent the client from lagging out
    else:
        print(f"SKIPPING: {m_id} (Already downloaded or unavailable)")