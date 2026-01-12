import tkinter as tk
import cv2
import numpy as np
import mss
import sys

# --- CONFIGURATION ---
TEMPLATE_PATH = 'leesin.png'
MATCH_CONFIDENCE = 0.75 # Slightly lower because resizing reduces accuracy slightly
# ---------------------

# 1. SETUP VISION
print("Loading template...")
try:
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print(f"Error: Could not load '{TEMPLATE_PATH}'. Make sure file exists!")
        sys.exit()
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    tH, tW = template_gray.shape[:2]
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# 2. SETUP GUI
root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)

window_width = 150
window_height = 80
screen_width = root.winfo_screenwidth()
x_pos = screen_width - window_width - 10
y_pos = 10

root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

status_label = tk.Label(root, text="SEARCHING", font=("Helvetica", 12, "bold"), bg="red", fg="white")
status_label.pack(fill="both", expand=True)

def close_app(event=None):
    root.destroy()
    sys.exit()
root.bind("<Escape>", close_app)

# 3. MULTI-SCALE SCANNING LOGIC
def scan_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[0] 
        screenshot = np.array(sct.grab(monitor))
    
    screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    found = None

    # Loop over the scales of the template
    # We check sizes from 50% of original to 150% of original
    for scale in np.linspace(0.5, 1.5, 10):
        
        # Resize the template according to the current scale
        resized_template = cv2.resize(template_gray, (int(tW * scale), int(tH * scale)))
        r = tW / float(resized_template.shape[1]) # Ratio

        # If the resized template is larger than the image, skip it
        if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
            continue

        # Apply template matching to find the template in the image
        result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        
        # Get the best match score for this specific scale
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # If this specific scale match is better than any previous scale, save it
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # Check the best match found across ALL scales
    if found is not None and found[0] >= MATCH_CONFIDENCE:
        status_label.config(bg="green", text="DETECTED!")
    else:
        status_label.config(bg="red", text="SEARCHING...")

    # Schedule next check
    root.after(100, scan_screen)

print("Smart tracker started. Press 'Esc' to close.")
scan_screen()
root.mainloop()