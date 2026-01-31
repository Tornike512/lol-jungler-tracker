"""
Action Execution Layer with Human-like Input Simulation
Implements safe, randomized mouse/keyboard control with anti-detection measures.
"""
import time
import platform
import random
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .config import action_cfg, safety_cfg

# Platform-specific imports
PLATFORM = platform.system()

if PLATFORM == "Windows":
    try:
        import ctypes
        from ctypes import wintypes

        # Windows input structures
        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        MOUSEEVENTF_RIGHTDOWN = 0x0008
        MOUSEEVENTF_RIGHTUP = 0x0010
        MOUSEEVENTF_ABSOLUTE = 0x8000

        KEYEVENTF_KEYUP = 0x0002

        # Virtual key codes
        VK_CODES = {
            'q': 0x51, 'w': 0x57, 'e': 0x45, 'r': 0x52,
            'd': 0x44, 'f': 0x46, 'a': 0x41, 's': 0x53,
            'b': 0x42, 'p': 0x50, 'tab': 0x09,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37,
            'ctrl': 0x11, 'shift': 0x10, 'alt': 0x12,
            'space': 0x20, 'spacebar': 0x20,
        }

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("union", INPUT_UNION)
            ]

        WINDOWS_API_AVAILABLE = True
    except ImportError:
        WINDOWS_API_AVAILABLE = False
        print("Warning: Windows API not available")
elif PLATFORM == "Linux":
    try:
        from pynput.mouse import Controller as MouseController, Button
        from pynput.keyboard import Controller as KeyboardController, Key
        PYNPUT_AVAILABLE = True
    except ImportError:
        PYNPUT_AVAILABLE = False
        print("Warning: pynput not available. Install with: pip install pynput")
else:
    print(f"Warning: Unsupported platform: {PLATFORM}")


@dataclass
class MouseState:
    """Current mouse state"""
    x: int
    y: int
    last_click_time: float = 0.0
    last_move_time: float = 0.0


@dataclass
class KeyboardState:
    """Current keyboard state"""
    last_key_time: float = 0.0
    pressed_keys: set = None

    def __post_init__(self):
        if self.pressed_keys is None:
            self.pressed_keys = set()


class BezierCurve:
    """Generate smooth Bezier curves for human-like mouse movement"""

    @staticmethod
    def cubic_bezier(
        start: Tuple[int, int],
        end: Tuple[int, int],
        num_points: int = 50,
        randomness: float = 0.3
    ) -> np.ndarray:
        """
        Generate a cubic Bezier curve from start to end.

        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            num_points: Number of points in the curve
            randomness: Amount of randomness in control points (0-1)

        Returns:
            Array of (x, y) points along the curve
        """
        x0, y0 = start
        x3, y3 = end

        # Calculate distance and angle
        dx = x3 - x0
        dy = y3 - y0
        distance = np.sqrt(dx**2 + dy**2)

        # Generate control points with randomness
        # Control point 1: 1/3 of the way with perpendicular offset
        t1 = 0.33
        offset1 = random.uniform(-distance * randomness, distance * randomness)
        x1 = x0 + t1 * dx + offset1 * (-dy / (distance + 1e-6))
        y1 = y0 + t1 * dy + offset1 * (dx / (distance + 1e-6))

        # Control point 2: 2/3 of the way with perpendicular offset
        t2 = 0.67
        offset2 = random.uniform(-distance * randomness, distance * randomness)
        x2 = x0 + t2 * dx + offset2 * (-dy / (distance + 1e-6))
        y2 = y0 + t2 * dy + offset2 * (dx / (distance + 1e-6))

        # Generate points along curve
        t = np.linspace(0, 1, num_points)

        # Cubic Bezier formula
        x = (1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3
        y = (1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3

        points = np.column_stack([x, y])
        return points.astype(int)


class InputController:
    """
    Cross-platform input controller with human-like behavior.
    Supports both Windows (SendInput API) and Linux (pynput).
    """

    def __init__(self):
        self.platform = PLATFORM
        self.mouse_state = MouseState(x=0, y=0)
        self.keyboard_state = KeyboardState()

        # Initialize platform-specific controllers
        if self.platform == "Windows":
            if not WINDOWS_API_AVAILABLE:
                raise RuntimeError("Windows API not available")
            self._init_windows()
        elif self.platform == "Linux":
            if not PYNPUT_AVAILABLE:
                raise RuntimeError("pynput not available. Install with: pip install pynput")
            self._init_linux()
        else:
            raise RuntimeError(f"Unsupported platform: {self.platform}")

        # APM tracking
        self.action_times = []
        self.max_apm_history = 60  # Track last 60 seconds

    def _init_windows(self):
        """Initialize Windows-specific input simulation"""
        # Get initial mouse position
        point = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        self.mouse_state.x = point.x
        self.mouse_state.y = point.y

        # Get screen dimensions for absolute positioning
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    def _init_linux(self):
        """Initialize Linux-specific input simulation"""
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        # Get initial mouse position
        self.mouse_state.x, self.mouse_state.y = self.mouse.position

    def get_current_apm(self) -> float:
        """Calculate current Actions Per Minute"""
        current_time = time.time()

        # Remove old actions (older than 60 seconds)
        self.action_times = [t for t in self.action_times if current_time - t < 60.0]

        # Calculate APM
        if len(self.action_times) == 0:
            return 0.0

        time_span = current_time - self.action_times[0]
        if time_span == 0:
            return 0.0

        apm = (len(self.action_times) / time_span) * 60.0
        return apm

    def should_throttle_action(self) -> bool:
        """Check if action should be throttled to maintain human-like APM"""
        current_apm = self.get_current_apm()
        return current_apm > action_cfg.MAX_APM

    def add_action_timestamp(self):
        """Record an action timestamp for APM tracking"""
        self.action_times.append(time.time())

    def apply_reaction_time(self):
        """Apply human-like reaction time delay"""
        # Log-normal distribution for reaction time
        reaction_time = np.random.lognormal(
            mean=np.log(action_cfg.AVG_REACTION_TIME_MS),
            sigma=np.log(1 + action_cfg.STD_REACTION_TIME_MS / action_cfg.AVG_REACTION_TIME_MS)
        )

        # Enforce minimum reaction time
        if safety_cfg.ENFORCE_MIN_REACTION_TIME:
            reaction_time = max(reaction_time, action_cfg.MIN_REACTION_TIME_MS)

        # Convert to seconds and sleep
        time.sleep(reaction_time / 1000.0)

    def move_mouse_to(
        self,
        target_x: int,
        target_y: int,
        use_bezier: bool = action_cfg.USE_BEZIER_CURVES,
        duration: Optional[float] = None
    ):
        """
        Move mouse to target position with human-like movement.

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            use_bezier: Whether to use Bezier curves for smooth movement
            duration: Optional movement duration in seconds (auto-calculated if None)
        """
        # Check APM throttling
        if self.should_throttle_action():
            time.sleep(random.uniform(0.05, 0.15))

        start_x, start_y = self.mouse_state.x, self.mouse_state.y

        # Calculate movement distance
        distance = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)

        if distance < 5:
            # Very short distance, just move directly
            self._set_mouse_position(target_x, target_y)
            return

        # Calculate movement duration if not provided
        if duration is None:
            # Speed varies based on distance (Fitts's law approximation)
            speed = random.uniform(action_cfg.MOUSE_SPEED_MIN, action_cfg.MOUSE_SPEED_MAX)
            duration = distance / (speed * 1000)  # Convert to seconds

        if use_bezier:
            # Generate Bezier curve
            num_points = max(int(duration * 100), 10)  # At least 10 points
            curve_points = BezierCurve.cubic_bezier(
                (start_x, start_y),
                (target_x, target_y),
                num_points=num_points,
                randomness=0.2
            )

            # Move along curve
            time_per_point = duration / num_points
            for point in curve_points:
                self._set_mouse_position(int(point[0]), int(point[1]))
                time.sleep(time_per_point)
        else:
            # Linear interpolation
            steps = max(int(duration * 100), 10)
            for i in range(steps + 1):
                t = i / steps
                x = int(start_x + (target_x - start_x) * t)
                y = int(start_y + (target_y - start_y) * t)
                self._set_mouse_position(x, y)
                time.sleep(duration / steps)

        self.add_action_timestamp()

    def _set_mouse_position(self, x: int, y: int):
        """Set mouse position (platform-specific)"""
        if self.platform == "Linux":
            self.mouse.position = (x, y)
        elif self.platform == "Windows":
            ctypes.windll.user32.SetCursorPos(x, y)

        self.mouse_state.x = x
        self.mouse_state.y = y
        self.mouse_state.last_move_time = time.time()

    def click_mouse(
        self,
        button: str = "left",
        clicks: int = 1,
        apply_debounce: bool = True
    ):
        """
        Click mouse button.

        Args:
            button: Button to click ("left" or "right")
            clicks: Number of clicks
            apply_debounce: Whether to apply debounce delay
        """
        # Check debounce
        if apply_debounce:
            time_since_last_click = time.time() - self.mouse_state.last_click_time
            if time_since_last_click < action_cfg.CLICK_DEBOUNCE_MS / 1000.0:
                return  # Too soon, skip click

        # Map button name
        if self.platform == "Linux":
            button_obj = Button.left if button == "left" else Button.right

            for _ in range(clicks):
                self.mouse.press(button_obj)
                time.sleep(random.uniform(0.05, 0.08))  # Human-like click duration
                self.mouse.release(button_obj)

                if clicks > 1:
                    time.sleep(random.uniform(0.08, 0.12))  # Delay between clicks

        elif self.platform == "Windows":
            if button == "left":
                down_flag = MOUSEEVENTF_LEFTDOWN
                up_flag = MOUSEEVENTF_LEFTUP
            else:
                down_flag = MOUSEEVENTF_RIGHTDOWN
                up_flag = MOUSEEVENTF_RIGHTUP

            for _ in range(clicks):
                # Mouse down
                input_down = INPUT()
                input_down.type = 0  # INPUT_MOUSE
                input_down.union.mi.dwFlags = down_flag
                ctypes.windll.user32.SendInput(1, ctypes.byref(input_down), ctypes.sizeof(INPUT))

                time.sleep(random.uniform(0.05, 0.08))

                # Mouse up
                input_up = INPUT()
                input_up.type = 0  # INPUT_MOUSE
                input_up.union.mi.dwFlags = up_flag
                ctypes.windll.user32.SendInput(1, ctypes.byref(input_up), ctypes.sizeof(INPUT))

                if clicks > 1:
                    time.sleep(random.uniform(0.08, 0.12))

        self.mouse_state.last_click_time = time.time()
        self.add_action_timestamp()

    def press_key(
        self,
        key: str,
        hold_duration: Optional[float] = None,
        apply_debounce: bool = True
    ):
        """
        Press a keyboard key.

        Args:
            key: Key to press (e.g., "q", "w", "ctrl+q", "1")
            hold_duration: Optional duration to hold key (seconds)
            apply_debounce: Whether to apply debounce delay
        """
        # Check debounce
        if apply_debounce:
            time_since_last_key = time.time() - self.keyboard_state.last_key_time
            if time_since_last_key < action_cfg.KEY_DEBOUNCE_MS / 1000.0:
                return

        # Parse key (handle combinations like "ctrl+q")
        keys_to_press = key.lower().split("+")

        if self.platform == "Linux":
            # Map special keys
            key_map = {
                "ctrl": Key.ctrl,
                "shift": Key.shift,
                "alt": Key.alt,
                "tab": Key.tab,
                "spacebar": Key.space,
                "space": Key.space,
                "b": "b",
            }

            # Press keys in order
            pressed = []
            for k in keys_to_press:
                key_obj = key_map.get(k, k)
                self.keyboard.press(key_obj)
                pressed.append(key_obj)
                time.sleep(random.uniform(0.01, 0.03))

            # Hold if specified
            if hold_duration:
                time.sleep(hold_duration)
            else:
                time.sleep(random.uniform(0.05, 0.08))

            # Release in reverse order
            for key_obj in reversed(pressed):
                self.keyboard.release(key_obj)
                time.sleep(random.uniform(0.01, 0.03))

        elif self.platform == "Windows":
            # Press keys in order
            pressed_vks = []
            for k in keys_to_press:
                vk = VK_CODES.get(k)
                if vk is None:
                    # Try single character
                    if len(k) == 1:
                        vk = ord(k.upper())
                    else:
                        continue

                # Key down
                input_down = INPUT()
                input_down.type = 1  # INPUT_KEYBOARD
                input_down.union.ki.wVk = vk
                input_down.union.ki.dwFlags = 0
                ctypes.windll.user32.SendInput(1, ctypes.byref(input_down), ctypes.sizeof(INPUT))

                pressed_vks.append(vk)
                time.sleep(random.uniform(0.01, 0.03))

            # Hold if specified
            if hold_duration:
                time.sleep(hold_duration)
            else:
                time.sleep(random.uniform(0.05, 0.08))

            # Release in reverse order
            for vk in reversed(pressed_vks):
                input_up = INPUT()
                input_up.type = 1  # INPUT_KEYBOARD
                input_up.union.ki.wVk = vk
                input_up.union.ki.dwFlags = KEYEVENTF_KEYUP
                ctypes.windll.user32.SendInput(1, ctypes.byref(input_up), ctypes.sizeof(INPUT))
                time.sleep(random.uniform(0.01, 0.03))

        self.keyboard_state.last_key_time = time.time()
        self.add_action_timestamp()

    def execute_action(self, action_dict: Dict[str, Any], screen_width: int = 1920, screen_height: int = 1080):
        """
        Execute a complete action from the RL agent.

        Args:
            action_dict: Action dictionary from agent containing continuous and discrete actions
            screen_width: Screen width for coordinate conversion
            screen_height: Screen height for coordinate conversion
        """
        # Apply reaction time delay
        self.apply_reaction_time()

        # 1. Move mouse (continuous action)
        continuous = action_dict["continuous"]
        mouse_x_norm = np.clip(continuous[0], -1, 1)
        mouse_y_norm = np.clip(continuous[1], -1, 1)

        # Convert normalized coordinates to screen coordinates
        # Assume normalized coords are relative to current position
        current_x, current_y = self.mouse_state.x, self.mouse_state.y

        # Move up to 200 pixels in any direction
        max_move = 200
        target_x = current_x + int(mouse_x_norm * max_move)
        target_y = current_y + int(mouse_y_norm * max_move)

        # Clamp to screen bounds
        target_x = np.clip(target_x, 0, screen_width - 1)
        target_y = np.clip(target_y, 0, screen_height - 1)

        self.move_mouse_to(target_x, target_y)

        # 2. Mouse click (discrete action)
        discrete_mouse = action_dict["discrete_mouse"]
        if discrete_mouse == 1:  # Left click
            self.click_mouse("left")
        elif discrete_mouse == 2:  # Right click
            self.click_mouse("right")

        # 3. Keyboard input (discrete action)
        discrete_keyboard = action_dict["discrete_keyboard"]
        keyboard_keys = action_cfg.KEYBOARD_KEYS

        if discrete_keyboard > 0 and discrete_keyboard < len(keyboard_keys):
            key = keyboard_keys[discrete_keyboard]
            if key != "none":
                self.press_key(key)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Input Controller")
    print("=" * 60)
    print(f"Platform: {PLATFORM}")

    if PLATFORM == "Linux" and not PYNPUT_AVAILABLE:
        print("ERROR: pynput not installed")
        print("Install with: pip install pynput")
        exit(1)

    # Create controller
    controller = InputController()

    print("\nTesting mouse movement...")
    print("Moving mouse in a square pattern (5 seconds)")

    try:
        # Get current position
        start_x, start_y = controller.mouse_state.x, controller.mouse_state.y
        print(f"Starting position: ({start_x}, {start_y})")

        # Move in a square
        size = 100
        positions = [
            (start_x + size, start_y),
            (start_x + size, start_y + size),
            (start_x, start_y + size),
            (start_x, start_y),
        ]

        for target_x, target_y in positions:
            print(f"Moving to: ({target_x}, {target_y})")
            controller.move_mouse_to(target_x, target_y)
            time.sleep(0.5)

        print(f"\nCurrent APM: {controller.get_current_apm():.1f}")

        print("\nTesting keyboard input...")
        print("Pressing 'q' key in 2 seconds...")
        time.sleep(2)
        controller.press_key("q")

        print("\nAll tests completed successfully!")

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
