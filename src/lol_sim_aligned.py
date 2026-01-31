"""
Aligned Simulation Environment for Sim-to-Real Transfer.

This environment matches the observation and action spaces of the real LoLEnvironment,
enabling policies trained here to transfer to the actual game.

Observation Space: 512-dim vector (same as vision pipeline output)
Action Space: Dict with continuous mouse + discrete mouse/keyboard (same as real env)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import random

# Import config to match real environment
try:
    from .config import vision_cfg, action_cfg, capture_cfg, reward_cfg
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: Config not available, using defaults")

# Pygame for rendering
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame_ce as pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False


# =============================================================================
# CONFIGURATION (matches real environment)
# =============================================================================

@dataclass
class SimConfig:
    """Simulation configuration matching real game parameters."""
    # Screen dimensions (for coordinate normalization)
    SCREEN_WIDTH: int = 1920
    SCREEN_HEIGHT: int = 1080

    # Game map (1000x1000 internal, mapped to screen coords for actions)
    MAP_SIZE: int = 1000

    # State vector dimension (must match vision_cfg.STATE_DIM)
    STATE_DIM: int = 512
    MAX_ENTITIES: int = 30  # Must match vision_cfg.MAX_ENTITIES_TRACKED

    # Garen stats
    GAREN_SPEED: float = 340.0
    GAREN_ATTACK_RANGE: float = 175.0
    GAREN_ATTACK_COOLDOWN: float = 0.625
    GAREN_BASE_AD: float = 66.0

    # Minion stats (with randomization ranges for domain randomization)
    MELEE_HP_BASE: float = 477.0
    MELEE_HP_VAR: float = 50.0  # +/- variation
    CASTER_HP_BASE: float = 296.0
    CASTER_HP_VAR: float = 30.0

    MINION_SPEED_BASE: float = 325.0
    MINION_SPEED_VAR: float = 25.0

    MELEE_DECAY_BASE: float = 25.0
    MELEE_DECAY_VAR: float = 5.0
    CASTER_DECAY_BASE: float = 20.0
    CASTER_DECAY_VAR: float = 4.0

    # Wave timing
    WAVE_INTERVAL_BASE: float = 30.0
    WAVE_INTERVAL_VAR: float = 3.0
    FIRST_WAVE_DELAY: float = 1.5

    # Gold values
    MELEE_GOLD: int = 21
    CASTER_GOLD: int = 14

    # Lane path (top lane)
    LANE_POINTS: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 100), (200, 100), (300, 100), (400, 100), (500, 100),
        (500, 200), (500, 300), (500, 400), (500, 500)
    ])

    # Spawn points
    GAREN_SPAWN: Tuple[int, int] = (850, 850)
    ENEMY_NEXUS: Tuple[int, int] = (50, 50)
    ALLY_NEXUS: Tuple[int, int] = (950, 950)

    # XP range
    XP_RANGE: float = 400.0


# =============================================================================
# GAME ENTITIES
# =============================================================================

class SimMinion:
    """Simulated minion entity."""

    # Class IDs matching vision_cfg.DEFAULT_CLASSES
    CLASS_IDS = {
        "minion_melee": 2,
        "minion_ranged": 3,
    }

    def __init__(self, x: float, y: float, minion_type: str, config: SimConfig, randomize: bool = True):
        self.x = x
        self.y = y
        self.minion_type = minion_type
        self.config = config
        self.alive = True

        # Apply domain randomization
        if minion_type == "melee":
            base_hp = config.MELEE_HP_BASE
            hp_var = config.MELEE_HP_VAR if randomize else 0
            base_decay = config.MELEE_DECAY_BASE
            decay_var = config.MELEE_DECAY_VAR if randomize else 0
            self.gold = config.MELEE_GOLD
            self.class_id = self.CLASS_IDS["minion_melee"]
        else:
            base_hp = config.CASTER_HP_BASE
            hp_var = config.CASTER_HP_VAR if randomize else 0
            base_decay = config.CASTER_DECAY_BASE
            decay_var = config.CASTER_DECAY_VAR if randomize else 0
            self.gold = config.CASTER_GOLD
            self.class_id = self.CLASS_IDS["minion_ranged"]

        # Randomize stats
        self.max_hp = base_hp + random.uniform(-hp_var, hp_var)
        self.hp = self.max_hp
        self.decay_rate = base_decay + random.uniform(-decay_var, decay_var)
        self.speed = config.MINION_SPEED_BASE + random.uniform(
            -config.MINION_SPEED_VAR, config.MINION_SPEED_VAR
        ) if randomize else config.MINION_SPEED_BASE

    def update(self, dt: float, target: Tuple[float, float]):
        """Update minion position and apply HP decay."""
        if not self.alive:
            return

        # Move toward target
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 10:
            move_dist = self.speed * dt
            self.x += (dx / dist) * move_dist
            self.y += (dy / dist) * move_dist

        # HP decay
        self.hp -= self.decay_rate * dt
        if self.hp <= 0:
            self.alive = False

    def take_damage(self, damage: float) -> bool:
        """Apply damage. Returns True if this killed the minion."""
        was_alive = self.alive and self.hp > 0
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False
            return was_alive
        return False

    @property
    def hp_percent(self) -> float:
        return max(0, self.hp / self.max_hp)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


class SimGaren:
    """Simulated Garen agent."""

    def __init__(self, x: float, y: float, config: SimConfig):
        self.x = x
        self.y = y
        self.config = config
        self.attack_cooldown = 0.0
        self.hp = 1.0  # Normalized HP (Garen doesn't take damage in this sim)
        self.mana = 1.0  # Garen is manaless but we track it for observation space

        # Ability cooldowns (not used in CS training but in observation)
        self.ability_cooldowns = {"Q": 0.0, "W": 0.0, "E": 0.0, "R": 0.0}

    def move_to_screen_pos(self, screen_x: float, screen_y: float, dt: float):
        """
        Move toward a screen position.
        screen_x, screen_y are in [-1, 1] normalized coordinates.
        """
        # Convert screen position to game world direction
        # In the real game, clicking on screen edge moves champion in that direction

        # Calculate target direction from screen position
        # Screen center (0, 0) = no movement
        # Screen edge = move in that direction

        move_magnitude = np.sqrt(screen_x**2 + screen_y**2)
        if move_magnitude < 0.1:  # Dead zone
            return

        # Normalize and apply speed
        move_magnitude = min(move_magnitude, 1.0)
        angle = np.arctan2(screen_y, screen_x)

        speed = self.config.GAREN_SPEED * move_magnitude
        dx = np.cos(angle) * speed * dt
        dy = np.sin(angle) * speed * dt

        # Update position (clamped to map)
        self.x = np.clip(self.x + dx, 0, self.config.MAP_SIZE)
        self.y = np.clip(self.y + dy, 0, self.config.MAP_SIZE)

    def update_cooldown(self, dt: float):
        """Update attack cooldown."""
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0, self.attack_cooldown - dt)

    def can_attack(self) -> bool:
        return self.attack_cooldown <= 0

    def attack(self, target: SimMinion) -> Tuple[bool, bool]:
        """
        Attempt to attack a target.
        Returns (hit, killed).
        """
        if not self.can_attack():
            return False, False

        dist = np.sqrt((target.x - self.x)**2 + (target.y - self.y)**2)
        if dist > self.config.GAREN_ATTACK_RANGE:
            return False, False

        self.attack_cooldown = self.config.GAREN_ATTACK_COOLDOWN
        killed = target.take_damage(self.config.GAREN_BASE_AD)
        return True, killed

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


# =============================================================================
# ALIGNED GYMNASIUM ENVIRONMENT
# =============================================================================

class LoLSimAligned(gym.Env):
    """
    Simulation environment aligned with the real LoLEnvironment.

    Observation Space: Box(512,) - matches vision pipeline output
    Action Space: Dict matching real environment:
        - continuous: Box(4,) [mouse_x, mouse_y, camera_x, camera_y]
        - discrete_mouse: Discrete(3) [none, left, right]
        - discrete_keyboard: Discrete(21)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        domain_randomization: bool = True,
        simulation_speed: float = 1.0,
        max_episode_steps: int = 3000,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.simulation_speed = simulation_speed
        self.max_episode_steps = max_episode_steps

        # Configuration
        self.config = SimConfig()
        if CONFIG_AVAILABLE:
            self.config.STATE_DIM = vision_cfg.STATE_DIM
            self.config.MAX_ENTITIES = vision_cfg.MAX_ENTITIES_TRACKED
            self.config.SCREEN_WIDTH = capture_cfg.SCREEN_WIDTH
            self.config.SCREEN_HEIGHT = capture_cfg.SCREEN_HEIGHT

        # Time step
        self.dt = 1.0 / 30.0 * simulation_speed

        # Observation space: 512-dim vector (matches real env)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.STATE_DIM,),
            dtype=np.float32
        )

        # Action space: matches real environment exactly
        self.action_space = spaces.Dict({
            "continuous": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),  # mouse_x, mouse_y, camera_x, camera_y
                dtype=np.float32
            ),
            "discrete_mouse": spaces.Discrete(3),  # none, left, right
            "discrete_keyboard": spaces.Discrete(21),  # keyboard actions
        })

        # Game state
        self.garen: Optional[SimGaren] = None
        self.enemy_minions: List[SimMinion] = []

        # Episode tracking
        self.current_step = 0
        self.episode_time = 0.0
        self.total_gold = 0
        self.last_hits = 0
        self.missed_attacks = 0
        self.minions_missed = 0

        # Wave spawning
        self.time_since_wave = 0.0
        self.waves_spawned = 0
        self.wave_interval = self.config.WAVE_INTERVAL_BASE

        # Mouse position tracking (for attack targeting)
        self.mouse_x = 0.0  # Normalized [-1, 1]
        self.mouse_y = 0.0

        # Rendering
        self.screen = None
        self.clock = None
        self.font = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Apply domain randomization to wave timing
        if self.domain_randomization:
            self.wave_interval = self.config.WAVE_INTERVAL_BASE + random.uniform(
                -self.config.WAVE_INTERVAL_VAR, self.config.WAVE_INTERVAL_VAR
            )

        # Reset Garen
        spawn = self.config.GAREN_SPAWN
        self.garen = SimGaren(spawn[0], spawn[1], self.config)

        # Clear minions
        self.enemy_minions = []

        # Reset tracking
        self.current_step = 0
        self.episode_time = 0.0
        self.total_gold = 0
        self.last_hits = 0
        self.missed_attacks = 0
        self.minions_missed = 0
        self.time_since_wave = self.wave_interval - self.config.FIRST_WAVE_DELAY
        self.waves_spawned = 0
        self.mouse_x = 0.0
        self.mouse_y = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        self.current_step += 1
        self.episode_time += self.dt
        reward = 0.0

        # --- Process continuous action (mouse movement) ---
        continuous = action["continuous"]
        self.mouse_x = np.clip(continuous[0], -1, 1)
        self.mouse_y = np.clip(continuous[1], -1, 1)
        # camera_x, camera_y (indices 2, 3) ignored in sim (camera always centered)

        # --- Process discrete mouse action ---
        mouse_action = action["discrete_mouse"]

        if mouse_action == 2:  # Right-click = move/attack
            # Move toward mouse position
            self.garen.move_to_screen_pos(self.mouse_x, self.mouse_y, self.dt)

            # Try to attack nearest minion in range
            attack_result = self._try_attack_nearest()
            if attack_result == "killed":
                reward += 10.0  # Last hit reward (matches reward_cfg.REWARD_CS_HIT * 10)
            elif attack_result == "hit":
                reward -= 0.1  # Hit but didn't kill
                self.missed_attacks += 1

        elif mouse_action == 1:  # Left-click (select/target)
            pass  # No effect in this sim

        # --- Update Garen cooldown ---
        self.garen.update_cooldown(self.dt)

        # --- Spawn minion waves ---
        self.time_since_wave += self.dt
        if self.time_since_wave >= self.wave_interval:
            self._spawn_wave()
            self.time_since_wave = 0.0
            self.waves_spawned += 1

        # --- Update minions ---
        minions_died_nearby = 0
        for minion in self.enemy_minions:
            was_alive = minion.alive
            minion.update(self.dt, self.config.ALLY_NEXUS)

            # Check if died from decay (not our attack)
            if was_alive and not minion.alive:
                dist = np.sqrt(
                    (minion.x - self.garen.x)**2 +
                    (minion.y - self.garen.y)**2
                )
                if dist <= self.config.XP_RANGE:
                    minions_died_nearby += 1

        # Penalty for minions we could have last-hit
        if minions_died_nearby > 0:
            reward -= 0.5 * minions_died_nearby
            self.minions_missed += minions_died_nearby

        # Remove dead minions
        self.enemy_minions = [m for m in self.enemy_minions if m.alive]

        # --- Lane presence reward ---
        lane_dist = self._distance_to_lane()
        normalized_dist = lane_dist / (self.config.MAP_SIZE / 2)
        reward -= 0.01 * normalized_dist  # Penalty for being far from lane

        # --- XP range reward ---
        in_xp_range = any(
            np.sqrt((m.x - self.garen.x)**2 + (m.y - self.garen.y)**2) <= self.config.XP_RANGE
            for m in self.enemy_minions
        )
        if in_xp_range:
            reward += 0.002  # Small reward for staying in XP range

        # --- Termination ---
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _try_attack_nearest(self) -> str:
        """
        Try to attack the nearest minion in range.
        Returns: "killed", "hit", or "miss"
        """
        if not self.garen.can_attack():
            return "miss"

        # Find nearest minion
        nearest = None
        nearest_dist = float('inf')

        for minion in self.enemy_minions:
            if not minion.alive:
                continue
            dist = np.sqrt(
                (minion.x - self.garen.x)**2 +
                (minion.y - self.garen.y)**2
            )
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = minion

        if nearest is None or nearest_dist > self.config.GAREN_ATTACK_RANGE:
            return "miss"

        hit, killed = self.garen.attack(nearest)
        if killed:
            self.total_gold += nearest.gold
            self.last_hits += 1
            return "killed"
        elif hit:
            return "hit"
        return "miss"

    def _spawn_wave(self):
        """Spawn a minion wave at enemy nexus."""
        spawn = self.config.ENEMY_NEXUS

        # 3 melee minions
        for _ in range(3):
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(-30, 30)
            minion = SimMinion(
                spawn[0] + offset_x,
                spawn[1] + offset_y,
                "melee",
                self.config,
                randomize=self.domain_randomization
            )
            self.enemy_minions.append(minion)

        # 3 caster minions
        for _ in range(3):
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(20, 50)
            minion = SimMinion(
                spawn[0] + offset_x,
                spawn[1] + offset_y,
                "caster",
                self.config,
                randomize=self.domain_randomization
            )
            self.enemy_minions.append(minion)

    def _get_observation(self) -> np.ndarray:
        """
        Build 512-dim observation vector matching real vision pipeline.

        Structure (matching StateVectorizer):
        - [0]: HP percent
        - [1]: Mana percent
        - [2-4]: Q, W, E cooldowns
        - [5]: R cooldown
        - [6]: Minimap enemies (normalized)
        - [7:217]: Entity features (30 entities * 7 features)
        - [217:512]: Padding zeros
        """
        obs = np.zeros(self.config.STATE_DIM, dtype=np.float32)

        # Player state
        obs[0] = self.garen.hp
        obs[1] = self.garen.mana
        obs[2] = self.garen.ability_cooldowns["Q"]
        obs[3] = self.garen.ability_cooldowns["W"]
        obs[4] = self.garen.ability_cooldowns["E"]
        obs[5] = self.garen.ability_cooldowns["R"]
        obs[6] = 0.0  # No enemies on minimap in sim

        # Entity features
        # Sort minions by distance
        minions_with_dist = []
        for m in self.enemy_minions:
            if m.alive:
                dist = np.sqrt(
                    (m.x - self.garen.x)**2 +
                    (m.y - self.garen.y)**2
                )
                minions_with_dist.append((m, dist))

        minions_with_dist.sort(key=lambda x: x[1])

        # Take up to MAX_ENTITIES
        for i, (minion, dist) in enumerate(minions_with_dist[:self.config.MAX_ENTITIES]):
            base_idx = 7 + i * 7

            # Relative position to Garen (normalized to [-1, 1] based on screen)
            rel_x = (minion.x - self.garen.x) / (self.config.SCREEN_WIDTH / 2)
            rel_y = (minion.y - self.garen.y) / (self.config.SCREEN_HEIGHT / 2)

            # Bounding box size (normalized) - approximate
            width = 50.0 / self.config.SCREEN_WIDTH
            height = 50.0 / self.config.SCREEN_HEIGHT

            # Normalized distance
            norm_dist = dist / self.config.MAP_SIZE

            obs[base_idx] = float(minion.class_id) / 10.0  # class_id normalized
            obs[base_idx + 1] = 0.9  # confidence (simulated)
            obs[base_idx + 2] = np.clip(rel_x, -1, 1)
            obs[base_idx + 3] = np.clip(rel_y, -1, 1)
            obs[base_idx + 4] = width
            obs[base_idx + 5] = height
            obs[base_idx + 6] = norm_dist

        # Additional useful features in padding area
        # Garen position (normalized)
        obs[220] = self.garen.x / self.config.MAP_SIZE
        obs[221] = self.garen.y / self.config.MAP_SIZE

        # Attack ready
        obs[222] = 1.0 if self.garen.can_attack() else 0.0

        # Time normalized
        obs[223] = min(1.0, self.episode_time / 200.0)

        # Lane distance normalized
        obs[224] = self._distance_to_lane() / (self.config.MAP_SIZE / 2)

        # Nearest minion HP (if any)
        if minions_with_dist:
            obs[225] = minions_with_dist[0][0].hp_percent

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        return {
            "step": self.current_step,
            "time": self.episode_time,
            "gold": self.total_gold,
            "last_hits": self.last_hits,
            "missed_attacks": self.missed_attacks,
            "minions_missed": self.minions_missed,
            "waves_spawned": self.waves_spawned,
            "enemy_minions": len(self.enemy_minions),
            "garen_position": self.garen.position,
            "cs_per_minute": (self.last_hits / max(1, self.episode_time)) * 60,
        }

    def _distance_to_lane(self) -> float:
        """Calculate distance from Garen to lane center."""
        min_dist = float('inf')
        for point in self.config.LANE_POINTS:
            dist = np.sqrt(
                (self.garen.x - point[0])**2 +
                (self.garen.y - point[1])**2
            )
            min_dist = min(min_dist, dist)
        return min_dist

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self):
        """Render the environment."""
        if self.render_mode is None or not PYGAME_AVAILABLE:
            return None

        if self.screen is None:
            self._init_render()

        # Clear
        self.screen.fill((34, 139, 34))

        # Draw lane
        self._draw_lane()

        # Draw nexuses
        self._draw_nexuses()

        # Draw minions
        for minion in self.enemy_minions:
            self._draw_minion(minion)

        # Draw Garen
        self._draw_garen()

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

    def _init_render(self):
        """Initialize Pygame."""
        pygame.init()
        pygame.display.set_caption("LoL Sim Aligned - Transfer Learning Environment")
        self.window_size = 600
        self.scale = self.window_size / self.config.MAP_SIZE
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 80))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def _draw_lane(self):
        points = [self._to_screen(p[0], p[1]) for p in self.config.LANE_POINTS]
        pygame.draw.lines(self.screen, (139, 119, 101), False, points,
                         int(150 * self.scale))

    def _draw_nexuses(self):
        blue_pos = self._to_screen(*self.config.ALLY_NEXUS)
        pygame.draw.circle(self.screen, (0, 100, 255), blue_pos, 20)
        red_pos = self._to_screen(*self.config.ENEMY_NEXUS)
        pygame.draw.circle(self.screen, (255, 50, 50), red_pos, 20)

    def _draw_minion(self, minion: SimMinion):
        x, y = self._to_screen(minion.x, minion.y)
        color = (200, 50, 50) if minion.minion_type == "melee" else (255, 100, 100)
        radius = 10 if minion.minion_type == "melee" else 8
        pygame.draw.circle(self.screen, color, (x, y), radius)

        # HP bar
        hp_w = radius * 2
        hp_h = 3
        hp_x = x - hp_w // 2
        hp_y = y - radius - 5
        pygame.draw.rect(self.screen, (60, 60, 60), (hp_x, hp_y, hp_w, hp_h))
        hp_fill = int(hp_w * minion.hp_percent)
        hp_color = (int(255 * (1 - minion.hp_percent)), int(255 * minion.hp_percent), 0)
        pygame.draw.rect(self.screen, hp_color, (hp_x, hp_y, hp_fill, hp_h))

    def _draw_garen(self):
        x, y = self._to_screen(self.garen.x, self.garen.y)
        pygame.draw.circle(self.screen, (255, 200, 0), (x, y), 15)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 15, 2)

        # Attack range
        range_r = int(self.config.GAREN_ATTACK_RANGE * self.scale)
        s = pygame.Surface((range_r * 2, range_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 0, 30), (range_r, range_r), range_r)
        self.screen.blit(s, (x - range_r, y - range_r))

        # Attack ready indicator
        indicator_color = (0, 255, 0) if self.garen.can_attack() else (255, 0, 0)
        pygame.draw.circle(self.screen, indicator_color, (x, y - 20), 5)

    def _draw_hud(self):
        hud_y = self.window_size + 5
        pygame.draw.rect(self.screen, (30, 30, 30), (0, self.window_size, self.window_size, 80))

        stats = [
            f"Time: {self.episode_time:.1f}s",
            f"Gold: {self.total_gold}",
            f"CS: {self.last_hits}",
            f"Missed: {self.minions_missed}",
            f"CS/min: {(self.last_hits / max(1, self.episode_time)) * 60:.1f}",
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (255, 255, 255))
            col = i % 3
            row = i // 3
            self.screen.blit(text, (10 + col * 200, hud_y + row * 25))

        # Domain randomization indicator
        dr_text = "Domain Randomization: ON" if self.domain_randomization else "OFF"
        text = self.font.render(dr_text, True, (150, 150, 150))
        self.screen.blit(text, (10, hud_y + 55))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# =============================================================================
# TESTING
# =============================================================================

def test_aligned_env():
    """Test the aligned environment."""
    print("=" * 60)
    print("Testing Aligned Simulation Environment")
    print("=" * 60)

    env = LoLSimAligned(render_mode="human", domain_randomization=True)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    total_reward = 0
    done = False
    step = 0

    try:
        while not done and step < 2000:
            # Sample action matching real env format
            action = {
                "continuous": env.action_space["continuous"].sample(),
                "discrete_mouse": env.action_space["discrete_mouse"].sample(),
                "discrete_keyboard": env.action_space["discrete_keyboard"].sample(),
            }

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            env.render()

            if step % 200 == 0:
                print(f"Step {step}: reward={reward:.3f}, total={total_reward:.2f}, "
                      f"CS={info['last_hits']}, gold={info['gold']}")

            step += 1

            if PYGAME_AVAILABLE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

    except KeyboardInterrupt:
        print("\nStopped")

    print(f"\nFinal: steps={step}, reward={total_reward:.2f}, CS={info['last_hits']}")
    env.close()


if __name__ == "__main__":
    test_aligned_env()
