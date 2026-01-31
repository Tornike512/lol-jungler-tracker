"""
Simulated League of Legends Top Lane Environment for Reinforcement Learning.

A 2D Gymnasium environment that simulates a simplified top lane last-hitting micro-game.
Designed for training an AI (playing as Garen) to walk to lane and last-hit minions for gold.

Features:
- 1000x1000 coordinate system with top lane path
- Garen agent with position, attack cooldown, and melee range
- Minion waves with HP decay (simulating allied minions attacking)
- Configurable rewards for last-hitting, lane presence, and survival
- Optional Pygame rendering for visualization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

# Try to import Pygame for rendering (supports both pygame and pygame-ce)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame_ce as pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False
        print("Pygame not available. Rendering disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MapConfig:
    """Map configuration constants."""
    WIDTH: int = 1000
    HEIGHT: int = 1000

    # Top lane path (diagonal from top-left to middle-right area)
    # Lane is defined by a center line and width
    LANE_START: Tuple[int, int] = (50, 50)  # Top-left (enemy nexus for blue)
    LANE_END: Tuple[int, int] = (950, 950)  # Bottom-right (ally nexus for blue)
    LANE_WIDTH: float = 150.0  # Width of the lane path

    # Nexus positions
    BLUE_NEXUS: Tuple[int, int] = (950, 950)  # Blue team nexus (bottom-right)
    RED_NEXUS: Tuple[int, int] = (50, 50)     # Red team nexus (top-left)

    # Top lane center line for rewards (runs along top-left edge)
    # Top lane goes from approximately (100, 100) to (500, 100) then (500, 500)
    TOP_LANE_POINTS: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 100), (200, 100), (300, 100), (400, 100), (500, 100),
        (500, 200), (500, 300), (500, 400), (500, 500)
    ])

    # Garen spawn position (near blue nexus, heading to top lane)
    GAREN_SPAWN: Tuple[int, int] = (850, 850)

    # XP range (distance to gain experience from minion deaths)
    XP_RANGE: float = 400.0


@dataclass
class GarenConfig:
    """Garen champion configuration."""
    BASE_MOVEMENT_SPEED: float = 340.0  # Units per second (Garen's actual base MS)
    ATTACK_RANGE: float = 175.0         # Melee attack range
    BASE_ATTACK_COOLDOWN: float = 0.625 # Seconds between attacks (1.6 AS at level 1)
    BASE_ATTACK_DAMAGE: float = 66.0    # Base AD at level 1


@dataclass
class MinionConfig:
    """Minion configuration."""
    # HP values
    MELEE_HP: float = 477.0
    CASTER_HP: float = 296.0

    # Movement speed
    MELEE_SPEED: float = 325.0
    CASTER_SPEED: float = 325.0

    # HP decay rate (simulates allied minions attacking)
    # In a real game, minions lose HP over ~15-20 seconds in lane
    MELEE_DECAY_RATE: float = 25.0   # HP per second
    CASTER_DECAY_RATE: float = 20.0  # HP per second

    # Spawn timing
    WAVE_INTERVAL: float = 30.0  # Seconds between waves
    FIRST_WAVE_TIME: float = 1.5 # Initial spawn delay

    # Minions per wave
    MELEE_PER_WAVE: int = 3
    CASTER_PER_WAVE: int = 3

    # Gold values
    MELEE_GOLD: int = 21
    CASTER_GOLD: int = 14


@dataclass
class RewardConfig:
    """Reward configuration."""
    LAST_HIT: float = 10.0           # Successful last hit
    MISSED_LAST_HIT: float = -0.1    # Attacked but didn't kill (bad timing)
    FAR_FROM_LANE: float = -0.01     # Per step penalty for being away from lane
    XP_RANGE_BONUS: float = 0.05     # Per second for staying in XP range
    MINION_DIED_NEARBY: float = -0.5 # Minion died but we didn't last hit it


# =============================================================================
# GAME ENTITIES
# =============================================================================

class Minion:
    """Represents a single minion."""

    def __init__(
        self,
        x: float,
        y: float,
        minion_type: str,  # "melee" or "caster"
        team: str,         # "red" (enemy) or "blue" (ally)
        config: MinionConfig
    ):
        self.x = x
        self.y = y
        self.minion_type = minion_type
        self.team = team
        self.config = config

        # Set HP based on type
        if minion_type == "melee":
            self.max_hp = config.MELEE_HP
            self.speed = config.MELEE_SPEED
            self.decay_rate = config.MELEE_DECAY_RATE
            self.gold_value = config.MELEE_GOLD
        else:
            self.max_hp = config.CASTER_HP
            self.speed = config.CASTER_SPEED
            self.decay_rate = config.CASTER_DECAY_RATE
            self.gold_value = config.CASTER_GOLD

        self.hp = self.max_hp
        self.alive = True

    def update(self, dt: float, target: Tuple[float, float]):
        """Update minion position and HP decay."""
        if not self.alive:
            return

        # Move toward target (enemy nexus)
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 10:  # Don't move if very close
            move_dist = self.speed * dt
            self.x += (dx / dist) * move_dist
            self.y += (dy / dist) * move_dist

        # HP decay (simulates allied minions attacking)
        self.hp -= self.decay_rate * dt

        if self.hp <= 0:
            self.alive = False

    def take_damage(self, damage: float) -> bool:
        """
        Apply damage to the minion.
        Returns True if this killed the minion (last hit).
        """
        was_alive = self.alive and self.hp > 0
        self.hp -= damage

        if self.hp <= 0:
            self.alive = False
            return was_alive  # Return True if we killed it
        return False

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def hp_percent(self) -> float:
        return max(0, self.hp / self.max_hp)


class Garen:
    """Represents the Garen agent."""

    def __init__(self, x: float, y: float, config: GarenConfig):
        self.x = x
        self.y = y
        self.config = config

        # Attack state
        self.attack_cooldown = 0.0  # Time until next attack is ready
        self.attack_damage = config.BASE_ATTACK_DAMAGE

    def move(self, angle: float, magnitude: float, dt: float):
        """
        Move Garen in a direction.

        Args:
            angle: Direction in radians (0 = right, pi/2 = up)
            magnitude: Movement speed multiplier (0-1)
            dt: Delta time in seconds
        """
        # Calculate movement
        speed = self.config.BASE_MOVEMENT_SPEED * np.clip(magnitude, 0, 1)
        dx = np.cos(angle) * speed * dt
        dy = np.sin(angle) * speed * dt

        # Update position (clamped to map bounds)
        self.x = np.clip(self.x + dx, 0, MapConfig.WIDTH)
        self.y = np.clip(self.y + dy, 0, MapConfig.HEIGHT)

    def update_cooldown(self, dt: float):
        """Update attack cooldown."""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
            self.attack_cooldown = max(0, self.attack_cooldown)

    def can_attack(self) -> bool:
        """Check if attack is ready."""
        return self.attack_cooldown <= 0

    def attack(self, target: Minion) -> Tuple[bool, bool]:
        """
        Attempt to attack a target minion.

        Returns:
            (hit, killed): Whether attack landed and whether it killed the target
        """
        if not self.can_attack():
            return False, False

        # Check range
        dist = np.sqrt((target.x - self.x)**2 + (target.y - self.y)**2)
        if dist > self.config.ATTACK_RANGE:
            return False, False

        # Attack hits
        self.attack_cooldown = self.config.BASE_ATTACK_COOLDOWN
        killed = target.take_damage(self.attack_damage)

        return True, killed

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class LoLTopLaneEnv(gym.Env):
    """
    Simulated League of Legends Top Lane Environment.

    Observation Space (14-dimensional):
        - Agent x, y position (2)
        - Agent attack ready (1)
        - 3 closest enemy minions: x, y, hp_percent for each (9)
        - Time in episode (1)
        - Distance to lane center (1)

    Action Space (Hybrid):
        - Continuous: [angle, magnitude] for movement
        - Discrete: Attack target index (0-5, or 6 for no attack)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        simulation_speed: float = 1.0,
        max_episode_steps: int = 3000,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.simulation_speed = simulation_speed
        self.max_episode_steps = max_episode_steps

        # Configuration
        self.map_config = MapConfig()
        self.garen_config = GarenConfig()
        self.minion_config = MinionConfig()
        self.reward_config = RewardConfig()

        # Time step (fixed at ~30Hz simulation)
        self.dt = 1.0 / 30.0 * simulation_speed

        # Observation space: 14-dimensional vector
        # [garen_x, garen_y, attack_ready,
        #  minion1_x, minion1_y, minion1_hp,
        #  minion2_x, minion2_y, minion2_hp,
        #  minion3_x, minion3_y, minion3_hp,
        #  time_normalized, lane_distance_normalized]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        # Action space: Dict with movement and attack
        self.action_space = spaces.Dict({
            "move": spaces.Box(
                low=np.array([-np.pi, 0.0], dtype=np.float32),  # angle, magnitude
                high=np.array([np.pi, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "attack": spaces.Discrete(7)  # 0-5: target minion, 6: no attack
        })

        # Initialize state
        self.garen: Optional[Garen] = None
        self.enemy_minions: List[Minion] = []
        self.ally_minions: List[Minion] = []  # For visual purposes

        # Episode tracking
        self.current_step = 0
        self.episode_time = 0.0
        self.total_gold = 0
        self.last_hits = 0
        self.missed_last_hits = 0
        self.minions_missed = 0  # Died without us last hitting
        self.time_in_xp_range = 0.0

        # Wave spawning
        self.time_since_last_wave = 0.0
        self.waves_spawned = 0

        # Pygame rendering
        self.screen = None
        self.clock = None
        self.font = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset Garen at spawn position
        spawn_x, spawn_y = self.map_config.GAREN_SPAWN
        self.garen = Garen(spawn_x, spawn_y, self.garen_config)

        # Clear minions
        self.enemy_minions = []
        self.ally_minions = []

        # Reset episode tracking
        self.current_step = 0
        self.episode_time = 0.0
        self.total_gold = 0
        self.last_hits = 0
        self.missed_last_hits = 0
        self.minions_missed = 0
        self.time_in_xp_range = 0.0

        # Reset wave spawning
        self.time_since_last_wave = self.minion_config.WAVE_INTERVAL - self.minion_config.FIRST_WAVE_TIME
        self.waves_spawned = 0

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        self.episode_time += self.dt

        reward = 0.0

        # --- Process movement action ---
        move_action = action["move"]
        angle = move_action[0]
        magnitude = move_action[1]
        self.garen.move(angle, magnitude, self.dt)

        # --- Update Garen cooldown ---
        self.garen.update_cooldown(self.dt)

        # --- Process attack action ---
        attack_idx = action["attack"]
        if attack_idx < 6:  # Trying to attack a minion
            closest_minions = self._get_closest_enemy_minions(6)
            if attack_idx < len(closest_minions):
                target = closest_minions[attack_idx]
                hit, killed = self.garen.attack(target)

                if hit:
                    if killed:
                        # Successful last hit!
                        reward += self.reward_config.LAST_HIT
                        self.total_gold += target.gold_value
                        self.last_hits += 1
                    else:
                        # Hit but didn't kill (bad timing)
                        reward += self.reward_config.MISSED_LAST_HIT
                        self.missed_last_hits += 1

        # --- Spawn minion waves ---
        self.time_since_last_wave += self.dt
        if self.time_since_last_wave >= self.minion_config.WAVE_INTERVAL:
            self._spawn_wave()
            self.time_since_last_wave = 0.0
            self.waves_spawned += 1

        # --- Update minions ---
        # Enemy minions move toward blue nexus
        for minion in self.enemy_minions:
            minion.update(self.dt, self.map_config.BLUE_NEXUS)

            # Check if minion died from decay (not from our attack)
            if not minion.alive:
                # Check if we were in XP range
                dist_to_minion = self._distance(self.garen.position, minion.position)
                if dist_to_minion <= self.map_config.XP_RANGE:
                    # We were in range but didn't last hit
                    reward += self.reward_config.MINION_DIED_NEARBY
                    self.minions_missed += 1

        # Remove dead minions
        self.enemy_minions = [m for m in self.enemy_minions if m.alive]

        # --- Lane presence reward ---
        lane_distance = self._distance_to_lane()
        normalized_lane_dist = lane_distance / (self.map_config.WIDTH / 2)
        reward += self.reward_config.FAR_FROM_LANE * normalized_lane_dist

        # --- XP range bonus ---
        in_xp_range = any(
            self._distance(self.garen.position, m.position) <= self.map_config.XP_RANGE
            for m in self.enemy_minions
        )
        if in_xp_range:
            self.time_in_xp_range += self.dt
            reward += self.reward_config.XP_RANGE_BONUS * self.dt

        # --- Check termination ---
        terminated = False  # Episode doesn't naturally terminate
        truncated = self.current_step >= self.max_episode_steps

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _spawn_wave(self):
        """Spawn a new minion wave."""
        # Spawn position near red nexus (enemy)
        spawn_x, spawn_y = self.map_config.RED_NEXUS

        # Add some spread to spawn positions
        for i in range(self.minion_config.MELEE_PER_WAVE):
            offset_x = np.random.uniform(-30, 30)
            offset_y = np.random.uniform(-30, 30)
            minion = Minion(
                spawn_x + offset_x,
                spawn_y + offset_y,
                "melee",
                "red",
                self.minion_config
            )
            self.enemy_minions.append(minion)

        for i in range(self.minion_config.CASTER_PER_WAVE):
            offset_x = np.random.uniform(-30, 30)
            offset_y = np.random.uniform(20, 50)  # Casters spawn slightly behind
            minion = Minion(
                spawn_x + offset_x,
                spawn_y + offset_y,
                "caster",
                "red",
                self.minion_config
            )
            self.enemy_minions.append(minion)

    def _get_closest_enemy_minions(self, n: int) -> List[Minion]:
        """Get the n closest enemy minions to Garen."""
        if not self.enemy_minions:
            return []

        # Calculate distances
        distances = [
            (m, self._distance(self.garen.position, m.position))
            for m in self.enemy_minions if m.alive
        ]

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        return [m for m, _ in distances[:n]]

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        obs = np.zeros(14, dtype=np.float32)

        # Garen position (normalized to 0-1)
        obs[0] = self.garen.x / self.map_config.WIDTH
        obs[1] = self.garen.y / self.map_config.HEIGHT

        # Attack ready (boolean as 0/1)
        obs[2] = 1.0 if self.garen.can_attack() else 0.0

        # 3 closest enemy minions
        closest = self._get_closest_enemy_minions(3)
        for i, minion in enumerate(closest):
            base_idx = 3 + i * 3
            obs[base_idx] = minion.x / self.map_config.WIDTH
            obs[base_idx + 1] = minion.y / self.map_config.HEIGHT
            obs[base_idx + 2] = minion.hp_percent

        # Fill with zeros if fewer than 3 minions (already zeros from initialization)

        # Time normalized (assuming max ~100 seconds useful)
        obs[12] = min(1.0, self.episode_time / 100.0)

        # Distance to lane center (normalized)
        lane_distance = self._distance_to_lane()
        obs[13] = min(1.0, lane_distance / (self.map_config.WIDTH / 2))

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Build the info dictionary."""
        return {
            "step": self.current_step,
            "time": self.episode_time,
            "gold": self.total_gold,
            "last_hits": self.last_hits,
            "missed_last_hits": self.missed_last_hits,
            "minions_missed": self.minions_missed,
            "waves_spawned": self.waves_spawned,
            "time_in_xp_range": self.time_in_xp_range,
            "garen_position": self.garen.position,
            "enemy_minions": len(self.enemy_minions),
            "cs_per_minute": (self.last_hits / max(1, self.episode_time)) * 60,
        }

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _distance_to_lane(self) -> float:
        """Calculate distance from Garen to the top lane center line."""
        # Find minimum distance to any lane point
        min_dist = float('inf')
        garen_pos = self.garen.position

        for point in self.map_config.TOP_LANE_POINTS:
            dist = self._distance(garen_pos, point)
            min_dist = min(min_dist, dist)

        return min_dist

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if not PYGAME_AVAILABLE:
            return None

        if self.screen is None:
            self._init_pygame()

        # Clear screen
        self.screen.fill((34, 139, 34))  # Green grass background

        # Draw lane path
        self._draw_lane()

        # Draw nexuses
        self._draw_nexuses()

        # Draw minions
        for minion in self.enemy_minions:
            self._draw_minion(minion)

        # Draw Garen
        self._draw_garen()

        # Draw attack range indicator
        self._draw_attack_range()

        # Draw HUD
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

        return None

    def _init_pygame(self):
        """Initialize Pygame for rendering."""
        pygame.init()
        pygame.display.set_caption("LoL Top Lane Simulator - Garen CS Training")

        # Use 600x600 window (scaled from 1000x1000 game coords)
        self.window_width = 600
        self.window_height = 600
        self.scale = self.window_width / self.map_config.WIDTH

        self.screen = pygame.display.set_mode((self.window_width, self.window_height + 80))  # Extra for HUD
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert game coordinates to screen coordinates."""
        return int(x * self.scale), int(y * self.scale)

    def _draw_lane(self):
        """Draw the top lane path."""
        # Draw lane as a series of connected rectangles
        points = [self._to_screen(p[0], p[1]) for p in self.map_config.TOP_LANE_POINTS]

        # Draw lane width indicator
        lane_color = (139, 119, 101)  # Brown path
        for i in range(len(points) - 1):
            pygame.draw.line(self.screen, lane_color, points[i], points[i+1],
                           int(self.map_config.LANE_WIDTH * self.scale * 0.5))

        # Draw lane center line
        pygame.draw.lines(self.screen, (100, 80, 60), False, points, 2)

    def _draw_nexuses(self):
        """Draw the nexus positions."""
        # Blue nexus (ally)
        blue_pos = self._to_screen(*self.map_config.BLUE_NEXUS)
        pygame.draw.circle(self.screen, (0, 100, 255), blue_pos, 20)
        pygame.draw.circle(self.screen, (255, 255, 255), blue_pos, 20, 2)

        # Red nexus (enemy)
        red_pos = self._to_screen(*self.map_config.RED_NEXUS)
        pygame.draw.circle(self.screen, (255, 50, 50), red_pos, 20)
        pygame.draw.circle(self.screen, (255, 255, 255), red_pos, 20, 2)

    def _draw_minion(self, minion: Minion):
        """Draw a single minion."""
        x, y = self._to_screen(minion.x, minion.y)

        # Color based on team and type
        if minion.team == "red":
            if minion.minion_type == "melee":
                color = (200, 50, 50)  # Red melee
                radius = 10
            else:
                color = (255, 100, 100)  # Red caster
                radius = 8
        else:
            if minion.minion_type == "melee":
                color = (50, 50, 200)  # Blue melee
                radius = 10
            else:
                color = (100, 100, 255)  # Blue caster
                radius = 8

        # Draw minion body
        pygame.draw.circle(self.screen, color, (x, y), radius)

        # Draw HP bar
        hp_bar_width = radius * 2
        hp_bar_height = 3
        hp_x = x - hp_bar_width // 2
        hp_y = y - radius - 5

        # Background (dark)
        pygame.draw.rect(self.screen, (60, 60, 60),
                        (hp_x, hp_y, hp_bar_width, hp_bar_height))

        # HP fill (green to red gradient based on HP)
        hp_percent = minion.hp_percent
        hp_color = (int(255 * (1 - hp_percent)), int(255 * hp_percent), 0)
        pygame.draw.rect(self.screen, hp_color,
                        (hp_x, hp_y, int(hp_bar_width * hp_percent), hp_bar_height))

    def _draw_garen(self):
        """Draw Garen (the agent)."""
        x, y = self._to_screen(self.garen.x, self.garen.y)

        # Draw Garen as a larger circle with sword icon
        color = (255, 200, 0)  # Gold color for Garen
        pygame.draw.circle(self.screen, color, (x, y), 15)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 15, 2)

        # Attack ready indicator
        if self.garen.can_attack():
            pygame.draw.circle(self.screen, (0, 255, 0), (x, y - 20), 5)
        else:
            # Cooldown indicator
            cd_percent = self.garen.attack_cooldown / self.garen.config.BASE_ATTACK_COOLDOWN
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y - 20), 5)
            pygame.draw.arc(self.screen, (0, 255, 0),
                          (x - 5, y - 25, 10, 10),
                          -np.pi/2, -np.pi/2 + (1 - cd_percent) * 2 * np.pi, 2)

    def _draw_attack_range(self):
        """Draw Garen's attack range."""
        x, y = self._to_screen(self.garen.x, self.garen.y)
        range_radius = int(self.garen.config.ATTACK_RANGE * self.scale)

        # Semi-transparent attack range circle
        s = pygame.Surface((range_radius * 2, range_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 0, 30), (range_radius, range_radius), range_radius)
        pygame.draw.circle(s, (255, 255, 0, 100), (range_radius, range_radius), range_radius, 1)
        self.screen.blit(s, (x - range_radius, y - range_radius))

    def _draw_hud(self):
        """Draw heads-up display with game stats."""
        hud_y = self.window_height + 5

        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (0, self.window_height, self.window_width, 80))

        # Stats
        stats = [
            f"Time: {self.episode_time:.1f}s",
            f"Gold: {self.total_gold}",
            f"CS: {self.last_hits}",
            f"Missed: {self.missed_last_hits}",
            f"Minions Lost: {self.minions_missed}",
            f"CS/min: {(self.last_hits / max(1, self.episode_time)) * 60:.1f}",
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (255, 255, 255))
            col = i % 3
            row = i // 3
            self.screen.blit(text, (10 + col * 200, hud_y + row * 25))

        # Wave info
        wave_text = f"Waves: {self.waves_spawned} | Enemy Minions: {len(self.enemy_minions)}"
        text = self.font.render(wave_text, True, (200, 200, 200))
        self.screen.blit(text, (10, hud_y + 55))

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# =============================================================================
# FLAT ACTION WRAPPER (for easier use with standard RL algorithms)
# =============================================================================

class LoLTopLaneFlatEnv(LoLTopLaneEnv):
    """
    Flat action space version of the environment.

    Action Space: Box(3,) - [angle, magnitude, attack_index_normalized]
    Where attack_index_normalized: 0-0.857 = attack minion 0-5, 0.857-1.0 = no attack
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Override action space with flat Box
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Convert flat action to dict and step."""
        # Parse flat action
        angle = action[0]
        magnitude = action[1]
        attack_normalized = action[2]

        # Convert attack to discrete index
        if attack_normalized < 0.857:
            attack_idx = int(attack_normalized * 7)  # 0-5
        else:
            attack_idx = 6  # No attack

        # Create dict action
        dict_action = {
            "move": np.array([angle, magnitude], dtype=np.float32),
            "attack": attack_idx
        }

        return super().step(dict_action)


# =============================================================================
# TESTING
# =============================================================================

def test_random_agent():
    """Test the environment with random actions."""
    print("=" * 60)
    print("LoL Top Lane Simulator - Random Agent Test")
    print("=" * 60)

    # Create environment
    env = LoLTopLaneEnv(render_mode="human", simulation_speed=2.0)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Run episode
    total_reward = 0
    done = False
    step = 0

    print("\nRunning random agent (close window to stop)...")

    try:
        while not done:
            # Sample random action
            action = {
                "move": env.action_space["move"].sample(),
                "attack": env.action_space["attack"].sample()
            }

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Render
            env.render()

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: reward={reward:.3f}, total={total_reward:.2f}, "
                      f"CS={info['last_hits']}, gold={info['gold']}")

            step += 1

            # Handle Pygame events (close window)
            if PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\n{'=' * 60}")
    print(f"Episode finished after {step} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final stats: {info}")

    env.close()


def test_flat_env():
    """Test the flat action space version."""
    print("=" * 60)
    print("LoL Top Lane Simulator - Flat Action Space Test")
    print("=" * 60)

    env = LoLTopLaneFlatEnv(render_mode=None)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:
            print(f"Step {i}: obs_shape={obs.shape}, reward={reward:.3f}")

    print(f"\nFinal info: {info}")
    env.close()
    print("Flat env test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--flat":
        test_flat_env()
    else:
        test_random_agent()
