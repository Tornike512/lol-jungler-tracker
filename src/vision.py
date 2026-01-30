"""
Computer Vision Pipeline with YOLO Object Detection
Handles object detection, OCR, and state vectorization for RL agent.
"""
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO detection will not work.")

from .config import vision_cfg, capture_cfg
from .capture import FrameData, RegionExtractor


@dataclass
class Detection:
    """Represents a single detected object"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center_x, center_y
    timestamp: float


@dataclass
class GameState:
    """Complete game state vector for RL agent"""
    # Timestamp
    timestamp: float

    # Player state
    player_hp_percent: float
    player_mana_percent: float
    ability_cooldowns: Dict[str, float]  # Q, W, E, R -> cooldown in seconds

    # Detections
    detections: List[Detection]

    # Processed features
    feature_vector: np.ndarray  # 512-dim state representation

    # Minimap information
    minimap_visible_enemies: int


class YOLODetector:
    """
    YOLO-based object detection for League of Legends entities.
    Detects champions, minions, turrets, skillshots, etc.
    """

    def __init__(
        self,
        model_path: str = vision_cfg.YOLO_MODEL,
        confidence: float = vision_cfg.YOLO_CONFIDENCE,
        iou_threshold: float = vision_cfg.YOLO_IOU_THRESHOLD,
        device: str = vision_cfg.YOLO_DEVICE,
    ):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLO detection")

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Class names (will be populated from model or config)
        self.class_names = vision_cfg.DEFAULT_CLASSES

        # Performance tracking
        self.inference_times = []

    def detect(self, image: np.ndarray, timestamp: float = None) -> List[Detection]:
        """
        Run YOLO detection on an image.

        Args:
            image: Input image (BGR format)
            timestamp: Optional timestamp for detection

        Returns:
            List of Detection objects
        """
        if timestamp is None:
            timestamp = time.time()

        start_time = time.time()

        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)

        # Keep only last 100 inference times
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)

                    # Get class and confidence
                    conf = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])

                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        timestamp=timestamp
                    )
                    detections.append(detection)

        return detections

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def visualize_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes on image for visualization.

        Args:
            image: Input image
            detections: List of detections

        Returns:
            Image with drawn boxes
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Choose color based on class
            if "enemy" in det.class_name:
                color = (0, 0, 255)  # Red for enemies
            elif "ally" in det.class_name:
                color = (0, 255, 0)  # Green for allies
            elif "player" in det.class_name:
                color = (255, 255, 0)  # Cyan for player
            else:
                color = (255, 0, 255)  # Magenta for other

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, label_size[1])
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            cv2.putText(
                output,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return output


class HUDParser:
    """
    Parse HUD elements using OCR and pixel analysis.
    Extracts health, mana, and ability cooldowns.
    """

    def __init__(self):
        self.extractor = RegionExtractor()

        # Color ranges for health/mana bars in HSV
        self.health_color_range = (
            np.array([30, 100, 100]),  # Lower bound (green)
            np.array([90, 255, 255])   # Upper bound
        )

        self.mana_color_range = (
            np.array([100, 100, 100]),  # Lower bound (blue)
            np.array([130, 255, 255])   # Upper bound
        )

    def parse_hud(self, hud_image: np.ndarray) -> Dict[str, Any]:
        """
        Parse HUD image to extract game state information.

        Args:
            hud_image: HUD region image

        Returns:
            Dictionary with parsed HUD data
        """
        result = {
            "hp_percent": 0.0,
            "mana_percent": 0.0,
            "ability_cooldowns": {"Q": 0.0, "W": 0.0, "E": 0.0, "R": 0.0}
        }

        # Extract health bar
        try:
            health_bar = self.extractor.extract_health_bar(hud_image)
            hp_percent = self.extractor.calculate_bar_percentage(
                health_bar, self.health_color_range
            )
            result["hp_percent"] = hp_percent
        except Exception as e:
            print(f"Error parsing health bar: {e}")

        # Extract mana bar
        try:
            mana_bar = self.extractor.extract_mana_bar(hud_image)
            mana_percent = self.extractor.calculate_bar_percentage(
                mana_bar, self.mana_color_range
            )
            result["mana_percent"] = mana_percent
        except Exception as e:
            print(f"Error parsing mana bar: {e}")

        # Extract ability cooldowns (placeholder - would need OCR or pixel analysis)
        # For now, return 0.0 (abilities ready)
        # TODO: Implement OCR or pixel-based cooldown detection

        return result


class StateVectorizer:
    """
    Converts raw game observations into a fixed-size state vector for the RL agent.
    """

    def __init__(self, state_dim: int = vision_cfg.STATE_DIM):
        self.state_dim = state_dim
        self.max_entities = vision_cfg.MAX_ENTITIES_TRACKED

    def vectorize(
        self,
        detections: List[Detection],
        hud_data: Dict[str, Any],
        minimap_enemies: int = 0,
        player_position: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Convert game state into a fixed-size feature vector.

        Args:
            detections: List of detected objects
            hud_data: Parsed HUD information
            minimap_enemies: Number of enemies visible on minimap
            player_position: Optional player position (x, y)

        Returns:
            State vector (numpy array)
        """
        # Initialize feature vector
        features = []

        # 1. Player state features (5 dims)
        features.extend([
            hud_data.get("hp_percent", 0.0),
            hud_data.get("mana_percent", 0.0),
            hud_data["ability_cooldowns"].get("Q", 0.0),
            hud_data["ability_cooldowns"].get("W", 0.0),
            hud_data["ability_cooldowns"].get("E", 0.0),
        ])

        # 2. Ultimate cooldown (1 dim)
        features.append(hud_data["ability_cooldowns"].get("R", 0.0))

        # 3. Minimap awareness (1 dim)
        features.append(float(minimap_enemies) / 5.0)  # Normalized

        # 4. Entity features (max_entities * 7 features per entity)
        # Each entity: [class_id, confidence, rel_x, rel_y, width, height, distance]
        entity_features = []

        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        sorted_detections = sorted_detections[:self.max_entities]

        if player_position is None:
            # Assume player is at center of screen
            player_x = capture_cfg.SCREEN_WIDTH // 2
            player_y = capture_cfg.SCREEN_HEIGHT // 2
        else:
            player_x, player_y = player_position

        for det in sorted_detections:
            x1, y1, x2, y2 = det.bbox
            center_x, center_y = det.center

            # Relative position to player (-1 to 1)
            rel_x = (center_x - player_x) / (capture_cfg.SCREEN_WIDTH / 2)
            rel_y = (center_y - player_y) / (capture_cfg.SCREEN_HEIGHT / 2)

            # Bounding box size (normalized)
            width = (x2 - x1) / capture_cfg.SCREEN_WIDTH
            height = (y2 - y1) / capture_cfg.SCREEN_HEIGHT

            # Distance to player (normalized)
            distance = np.sqrt(rel_x**2 + rel_y**2)

            entity_features.extend([
                float(det.class_id) / 10.0,  # Normalized class ID
                det.confidence,
                rel_x,
                rel_y,
                width,
                height,
                distance
            ])

        # Pad with zeros if fewer entities than max
        entities_found = len(sorted_detections)
        padding_needed = (self.max_entities - entities_found) * 7
        entity_features.extend([0.0] * padding_needed)

        features.extend(entity_features)

        # Convert to numpy array
        state_vector = np.array(features, dtype=np.float32)

        # Ensure correct size (pad or truncate if needed)
        if len(state_vector) < self.state_dim:
            state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]

        return state_vector


class VisionPipeline:
    """
    Complete vision pipeline that orchestrates detection, HUD parsing, and state vectorization.
    """

    def __init__(
        self,
        model_path: str = vision_cfg.YOLO_MODEL,
        use_yolo: bool = True,
    ):
        self.use_yolo = use_yolo and YOLO_AVAILABLE

        # Initialize components
        if self.use_yolo:
            self.detector = YOLODetector(model_path=model_path)
        else:
            self.detector = None
            print("Warning: YOLO detection disabled")

        self.hud_parser = HUDParser()
        self.vectorizer = StateVectorizer()

        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0

    def process_frame(self, frame_data: FrameData) -> GameState:
        """
        Process a captured frame and extract game state.

        Args:
            frame_data: Captured frame with all regions

        Returns:
            GameState object with all parsed information
        """
        start_time = time.time()

        # 1. Run object detection on main viewport
        detections = []
        if self.use_yolo and "main" in frame_data.regions:
            detections = self.detector.detect(
                frame_data.regions["main"],
                timestamp=frame_data.timestamp
            )

        # 2. Parse HUD
        hud_data = {"hp_percent": 1.0, "mana_percent": 1.0, "ability_cooldowns": {"Q": 0.0, "W": 0.0, "E": 0.0, "R": 0.0}}
        if "hud" in frame_data.regions:
            hud_data = self.hud_parser.parse_hud(frame_data.regions["hud"])

        # 3. Count enemies on minimap (placeholder - would need minimap template matching)
        minimap_enemies = 0

        # 4. Vectorize state
        feature_vector = self.vectorizer.vectorize(
            detections=detections,
            hud_data=hud_data,
            minimap_enemies=minimap_enemies
        )

        # Create game state
        game_state = GameState(
            timestamp=frame_data.timestamp,
            player_hp_percent=hud_data["hp_percent"],
            player_mana_percent=hud_data["mana_percent"],
            ability_cooldowns=hud_data["ability_cooldowns"],
            detections=detections,
            feature_vector=feature_vector,
            minimap_visible_enemies=minimap_enemies
        )

        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.total_frames_processed += 1

        return game_state

    def get_performance_stats(self) -> Dict[str, float]:
        """Get vision pipeline performance statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_frames_processed
            if self.total_frames_processed > 0
            else 0.0
        )

        stats = {
            "avg_processing_time_ms": avg_processing_time * 1000,
            "frames_processed": self.total_frames_processed,
        }

        if self.use_yolo and self.detector:
            stats["avg_yolo_inference_ms"] = self.detector.get_avg_inference_time()

        return stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from .capture import ScreenCapture
    import sys

    print("Testing Vision Pipeline")
    print("=" * 60)

    # Check if YOLO is available
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    # Create vision pipeline (will download yolov8n.pt if not present)
    print("Initializing vision pipeline...")
    vision = VisionPipeline(use_yolo=True)

    # Create screen capture
    capture = ScreenCapture(target_fps=30)  # Lower FPS for testing

    print("Starting capture...")
    capture.start()

    try:
        time.sleep(1)  # Wait for capture to start
        print("Processing frames for 10 seconds...")
        print("Press Ctrl+C to stop")

        start_time = time.time()
        while time.time() - start_time < 10.0:
            frame = capture.get_latest_frame()
            if frame is not None:
                # Process frame
                game_state = vision.process_frame(frame)

                # Print stats every second
                if int(time.time() - start_time) % 1 == 0:
                    print(f"\nTimestamp: {game_state.timestamp:.2f}")
                    print(f"HP: {game_state.player_hp_percent:.2%}")
                    print(f"Detections: {len(game_state.detections)}")
                    print(f"State vector shape: {game_state.feature_vector.shape}")
                    print(f"Vision stats: {vision.get_performance_stats()}")

                    # Visualize detections if any
                    if game_state.detections and "main" in frame.regions:
                        vis_img = vision.detector.visualize_detections(
                            frame.regions["main"],
                            game_state.detections
                        )
                        cv2.imshow("Detections", vis_img)
                        cv2.waitKey(1)

            time.sleep(0.03)  # ~30 FPS processing

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        capture.stop()
        cv2.destroyAllWindows()

    print("\nFinal stats:")
    print(vision.get_performance_stats())
