import tkinter as tk
import cv2
import numpy as np
import mss
import sys
import requests
import os

# --- CONFIGURATION ---
TEMPLATE_NAME = 'champion_circle.png'
TARGET_SIZE = 20  # Keep this small for minimaps
MATCH_CONFIDENCE = 0.95 
# ---------------------

def create_circular_mask(h, w):
    """Creates a boolean mask with a white circle in the middle"""
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def download_and_circular_crop(champion_name):
    print(f"Downloading '{champion_name}' and cropping to circle...")
    
    try:
        version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        r = requests.get(version_url)
        versions = r.json()
        latest_version = versions[0]
    except Exception as e:
        print(f"Error fetching version: {e}")
        return False

    formatted_name = "".join([word.capitalize() for word in champion_name.split()])
    formatted_name = formatted_name.replace("'", "").replace(".", "")
    image_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/img/champion/{formatted_name}.png"
    
    try:
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            print(f"Error: Champion not found.")
            return False
        
        nparr = np.frombuffer(img_response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Resize to target size
        resized_img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        # 2. Create the Circular Mask
        h, w = resized_img.shape[:2]
        mask = create_circular_mask(h, w)
        
        # 3. Apply Mask (Make corners black/transparent)
        # We set pixels outside the circle to 0 (black)
        circular_img = resized_img.copy()
        circular_img[~mask] = 0
        
        # 4. Save the circular template
        cv2.imwrite(TEMPLATE_NAME, circular_img)
        print(f"Successfully created circular template '{TEMPLATE_NAME}'")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# --- MAIN PROGRAM ---
def main():
    champion_input = input("Enter Champion Name (e.g., 'Lee Sin'): ")
    
    if not download_and_circular_crop(champion_input):
        sys.exit()

    # Setup Vision
    try:
        template = cv2.imread(TEMPLATE_NAME, cv2.IMREAD_COLOR)
        if template is None:
            print("Error loading image.")
            sys.exit()
        
        # To use masks, we usually work in Color (BGR), but matching is faster in Grayscale.
        # However, standard matchTemplate with masks is tricky in OpenCV Python.
        # We will convert to grayscale for the mask.
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Create a grayscale mask (255 for circle, 0 for background)
        h, w = template_gray.shape[:2]
        mask_bool = create_circular_mask(h, w)
        template_mask = np.zeros((h, w), dtype=np.uint8)
        template_mask[mask_bool] = 255

    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    # Setup GUI
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

    # Scanning Loop
    def scan_screen():
        with mss.mss() as sct:
            monitor = sct.monitors[0] 
            screenshot = np.array(sct.grab(monitor))
        
        screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        found = None

        # Note: We use TM_CCORR_NORMED because it supports MASKS.
        # TM_CCOEFF_NORMED does not support masks in standard OpenCV Python builds.
        
        for scale in np.linspace(1.0, 2.5, 15):
            # Resize template and mask
            resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
            resized_mask = cv2.resize(template_mask, (int(w * scale), int(h * scale)))
            
            if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
                continue

            # MATCH WITH MASK
            result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCORR_NORMED, mask=resized_mask)
            
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, scale)

        # Update GUI
        if found is not None and found[0] >= MATCH_CONFIDENCE:
            status_label.config(bg="green", text=f"DETECTED!\n{champion_input}")
        else:
            status_label.config(bg="red", text=f"SEARCHING\n{champion_input}")

        root.after(50, scan_screen)

    print(f"Tracking {champion_input} (Circular Mask)...")
    scan_screen()
    root.mainloop()

if __name__ == "__main__":
    main()