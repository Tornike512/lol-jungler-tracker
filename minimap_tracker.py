"""
Minimap Position Tracker
========================
Uses screen capture to detect player position on the minimap.
"""
import mss
import cv2
import numpy as np
import time

# Minimap settings for 1920x1080 resolution with default HUD scale
MINIMAP_LEFT = 1647
MINIMAP_TOP = 817
MINIMAP_SIZE = 267
MAP_GAME_SIZE = 15000  # League map size in game units

# Colors for detection (BGR format for OpenCV)
# The player indicator has a cyan/white border
PLAYER_BORDER_LOW = np.array([180, 180, 180])  # Light gray/white
PLAYER_BORDER_HIGH = np.array([255, 255, 255])  # Pure white

# Allied champions are typically cyan/teal colored on minimap
ALLY_COLOR_LOW = np.array([180, 150, 0])   # Cyan-ish (BGR)
ALLY_COLOR_HIGH = np.array([255, 255, 100])


class MinimapTracker:
    def __init__(self, minimap_left=MINIMAP_LEFT, minimap_top=MINIMAP_TOP,
                 minimap_size=MINIMAP_SIZE):
        self.sct = mss.mss()
        self.minimap_region = {
            "left": minimap_left,
            "top": minimap_top,
            "width": minimap_size,
            "height": minimap_size
        }
        self.minimap_size = minimap_size
        self.last_position = None

    def capture_minimap(self):
        """Capture the minimap region of the screen."""
        screenshot = self.sct.grab(self.minimap_region)
        # Convert to numpy array (BGRA format)
        img = np.array(screenshot)
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def find_player_position(self, img):
        """
        Find the player's position on the minimap.
        The player has a distinctive cyan/teal arrow pointing to their icon.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Method 1: Look for the cyan/teal player arrow indicator
        # The self-indicator in League is a bright cyan/teal color
        lower_cyan = np.array([85, 150, 150])
        upper_cyan = np.array([105, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

        # Find contours in cyan mask
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter by size - player indicator is medium-sized
            valid_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]
            if valid_contours:
                # Get the one closest to where we expect player to be
                # (prioritize consistency with last position)
                if self.last_position:
                    last_minimap_x = (self.last_position[0] / MAP_GAME_SIZE) * self.minimap_size
                    last_minimap_y = self.minimap_size - (self.last_position[1] / MAP_GAME_SIZE) * self.minimap_size

                    def distance_to_last(contour):
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            return (cx - last_minimap_x)**2 + (cy - last_minimap_y)**2
                        return float('inf')

                    valid_contours.sort(key=distance_to_last)

                best = valid_contours[0]
                M = cv2.moments(best)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy

        # Method 2: Look for blue/allied champion icon color
        # Allied champions show as blue-ish on minimap
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = [c for c in contours if 30 < cv2.contourArea(c) < 600]
            if valid_contours:
                # Get largest blue blob (likely player icon)
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy

        # Method 3: Look for bright white border (player indicator has white outline)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, white_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find circular contours (player indicator is round)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if 30 < area < 400:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.4:
                            M = cv2.moments(contour)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                return cx, cy

        return None

    def minimap_to_game_coords(self, minimap_x, minimap_y):
        """Convert minimap pixel coordinates to game map coordinates."""
        # Minimap (0,0) is top-left which corresponds to top-left of game map
        # In League, (0,0) is bottom-left, so we need to flip Y
        game_x = (minimap_x / self.minimap_size) * MAP_GAME_SIZE
        game_y = ((self.minimap_size - minimap_y) / self.minimap_size) * MAP_GAME_SIZE
        return int(game_x), int(game_y)

    def get_player_position(self):
        """Get the player's position in game coordinates with smoothing."""
        img = self.capture_minimap()
        minimap_pos = self.find_player_position(img)

        if minimap_pos:
            game_x, game_y = self.minimap_to_game_coords(minimap_pos[0], minimap_pos[1])

            # Validate: if position jumped too far (>3000 units), likely bad detection
            # Champions can't teleport that far in 1 second normally
            if self.last_position:
                dist = ((game_x - self.last_position[0])**2 +
                        (game_y - self.last_position[1])**2)**0.5
                if dist > 3000:
                    # Suspicious jump - keep last position
                    return self.last_position[0], self.last_position[1], None

            self.last_position = (game_x, game_y)
            return game_x, game_y, minimap_pos

        # Return last known position if detection fails
        if self.last_position:
            return self.last_position[0], self.last_position[1], None

        # Default to top lane starting position
        return 4500, 11000, None

    def debug_capture(self, save_path="minimap_debug.png"):
        """Capture and save minimap with detected position marked."""
        img = self.capture_minimap()
        debug_img = img.copy()

        # Show cyan detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_cyan = np.array([85, 150, 150])
        upper_cyan = np.array([105, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

        # Draw all cyan contours in yellow
        contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 255), 1)

        # Find and mark player position
        minimap_pos = self.find_player_position(img)

        if minimap_pos:
            # Draw circle at detected position
            cv2.circle(debug_img, minimap_pos, 12, (0, 255, 0), 2)
            cv2.circle(debug_img, minimap_pos, 3, (0, 255, 0), -1)
            game_x, game_y = self.minimap_to_game_coords(minimap_pos[0], minimap_pos[1])
            cv2.putText(debug_img, f"({game_x}, {game_y})", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(debug_img, "Position not found", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imwrite(save_path, debug_img)

        # Also save the cyan mask for debugging
        cv2.imwrite(save_path.replace('.png', '_cyan.png'), cyan_mask)

        return debug_img, minimap_pos


def test_tracker():
    """Test the minimap tracker."""
    print("=" * 50)
    print("MINIMAP TRACKER TEST")
    print("=" * 50)
    print("\nMake sure League is running and visible!")
    print("Press Ctrl+C to stop.\n")

    tracker = MinimapTracker()

    # Save a debug image first
    print("Saving debug image to minimap_debug.png...")
    tracker.debug_capture()
    print("Check the image to see if detection is working.\n")

    print("Starting live tracking...")
    try:
        while True:
            game_x, game_y, minimap_pos = tracker.get_player_position()

            if minimap_pos:
                print(f"Position: ({game_x:5d}, {game_y:5d}) | Minimap pixel: {minimap_pos}", end='\r')
            else:
                print(f"Position: ({game_x:5d}, {game_y:5d}) | Detection failed, using last known", end='\r')

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    test_tracker()
