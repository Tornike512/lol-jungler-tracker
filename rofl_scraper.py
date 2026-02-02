import requests
import base64
import os
import time

# 1. Get the LCU Port and Password from the 'lockfile'
def get_lcu_credentials():
    lockfile_path = r"C:\Riot Games\League of Legends\lockfile"
    if not os.path.isfile(lockfile_path):
        return None, None
    with open(lockfile_path, 'r') as f:
        data = f.read().split(':')
        port = data[2]
        password = data[3]
    return port, password

# 2. Command the client to download a specific Match ID
def download_replay(match_id):
    port, password = get_lcu_credentials()
    if not port:
        print("League Client not found. Open the client first!")
        return

    # Encode credentials for Basic Auth
    auth = base64.b64encode(f"riot:{password}".encode()).decode()
    url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{match_id}/download/graceful"
    
    headers = {
        "Authorization": f"Basic {auth}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Payload is simply the Match ID
    response = requests.post(url, headers=headers, json={"gameId": match_id}, verify=False)
    
    if response.status_code == 204 or response.status_code == 200:
        print(f"Successfully started download for {match_id}")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

# Example Usage: List of Challenger Match IDs
match_list = [512345678, 512345679, 512345680]
for m_id in match_list:
    download_replay(m_id)
    time.sleep(5) # Give the client a moment between requests