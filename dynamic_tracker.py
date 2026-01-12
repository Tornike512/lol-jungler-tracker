import tkinter as tk
import cv2
import numpy as np
import mss
import sys
import requests
import os

# --- CONFIGURATION ---
TEMPLATE_NAME = 'champion.png'  # The file we will download
MATCH_CONFIDENCE = 0.75
# ---------------------

def download_champion_image(champion_name):
    print(f"Searching for '{champion_name}' on Data Dragon...")
    
    # 1. Get the latest League of Legends version
    try:
        version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        r = requests.get(version_url)
        versions = r.json()
        latest_version = versions[0]
    except Exception as e:
        print(f"Error fetching game version: {e}")
        return False

    # 2. Format the champion name for the URL
    # Riot uses TitleCase and removes spaces (e.g., "Lee Sin" -> "LeeSin")
    # Note: This works for 99% of champions. Special cases like "Wukong" or "Cho'Gath" might need exact typing.
    formatted_name = "".join([word.capitalize() for word in champion_name.split()])
    formatted_name = formatted_name.replace("'", "").replace(".", "")
    
    # 3. Construct the URL
    image_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/img/champion/{formatted_name}.png"
    
    print(f"Trying to download: {image_url}")

    # 4. Download the image
    try:
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(TEMPLATE_NAME, 'wb') as f:
                f.write(img_response.content)
            print("Download successful!")
            return True
        else:
            print(f"Error: Champion '{champion_name}' not found (Status: {img_response.status_code}).")
            print("Hint: Try the exact name, e.g., 'Aurelion Sol' or 'Nunu & Willump'")
            return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# --- MAIN PROGRAM ---
def main():
    # 1. Ask User
    champion_input = input("Enter Champion Name to track (e.g., 'Lee Sin', 'Ahri'): ")
    
    # 2. Download
    if not download_champion_image(champion_input):
        sys.exit()

    # 3. Setup Vision (Load the downloaded image)
    try:
        template = cv2.imread(TEMPLATE_NAME)
        if template is None:
            print("Error: Failed to load the downloaded image.")
            sys.exit()
        
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        tH, tW = template_gray.shape[:2]
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    # 4. Setup GUI
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    window_width = 150
    window_height = 80
    screen_width = root.winfo_screenwidth()
    x_pos = screen_width - window_width - 10
    y_pos = 10

    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

    status_label = tk.Label(root, text=f"SEARCHING\n{champion_input}", font=("Helvetica", 10, "bold"), bg="red", fg="white")
    status_label.pack(fill="both", expand=True)

    def close_app(event=None):
        root.destroy()
        sys.exit()
    root.bind("<Escape>", close_app)

    # 5. Multi-Scale Scanning Loop
    def scan_screen():
        with mss.mss() as sct:
            monitor = sct.monitors[0] 
            screenshot = np.array(sct.grab(monitor))
        
        screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        found = None

        # Scan from 50% size to 150% size
        for scale in np.linspace(0.5, 1.5, 10):
            resized_template = cv2.resize(template_gray, (int(tW * scale), int(tH * scale)))
            r = tW / float(resized_template.shape[1])

            if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
                continue

            result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # Update GUI
        if found is not None and found[0] >= MATCH_CONFIDENCE:
            status_label.config(bg="green", text=f"DETECTED!\n{champion_input}")
        else:
            status_label.config(bg="red", text=f"SEARCHING\n{champion_input}")

        root.after(100, scan_screen)

    print(f"Tracking {champion_input}... Press 'Esc' to quit.")
    scan_screen()
    root.mainloop()

if __name__ == "__main__":
    main()