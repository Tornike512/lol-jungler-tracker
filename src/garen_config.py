"""
Garen-specific configuration and mechanics.
This file defines Garen's abilities, combos, and training objectives.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GarenAbilities:
    """Garen's ability specifications"""

    # Q - Decisive Strike
    Q_COOLDOWN: float = 8.0  # seconds
    Q_RANGE: float = 125.0  # auto-attack range
    Q_SPEED_BOOST: float = 0.30  # 30% movement speed
    Q_SILENCE_DURATION: float = 1.5

    # W - Courage (Passive + Active)
    W_COOLDOWN: float = 23.0
    W_DURATION: float = 2.0
    W_DAMAGE_REDUCTION: float = 0.30  # 30% damage reduction

    # E - Judgment
    E_COOLDOWN: float = 9.0
    E_DURATION: float = 3.0
    E_RADIUS: float = 325.0
    E_CANCEL_MOVEMENT: bool = False  # Can move while spinning

    # R - Demacian Justice
    R_COOLDOWN_EARLY: float = 120.0
    R_COOLDOWN_LATE: float = 80.0  # at level 16
    R_RANGE: float = 400.0
    R_EXECUTE_THRESHOLD: float = 0.25  # Use when enemy <25% HP


@dataclass
class GarenCombos:
    """Garen combo patterns"""

    # Basic trade combo
    TRADE_COMBO: List[str] = None

    # All-in combo
    ALL_IN_COMBO: List[str] = None

    # Passive farming (healing)
    PASSIVE_REGEN: List[str] = None

    def __post_init__(self):
        # Q -> AA -> E (basic trade)
        self.TRADE_COMBO = ["q", "left_click", "e"]

        # Q -> AA -> E -> Ignite -> R (all-in)
        self.ALL_IN_COMBO = ["q", "left_click", "e", "d", "r"]

        # W -> Back off (take reduced damage while passive regens)
        self.PASSIVE_REGEN = ["w", "move_back"]


@dataclass
class GarenRewards:
    """Garen-specific reward shaping"""

    # CS rewards (Garen should focus on farming safely)
    REWARD_CS: float = 1.5  # Higher than default (Garen is a scaling champ)
    REWARD_CANNON: float = 3.0

    # Combat rewards
    REWARD_Q_AUTO_COMBO: float = 2.0  # Successfully land Q -> AA
    REWARD_E_MULTI_HIT: float = 1.0  # Hit 3+ minions/champions with E
    REWARD_R_EXECUTE: float = 10.0  # Kill with ultimate
    REWARD_R_WASTE: float = -5.0  # Use R on full HP enemy

    # Survival rewards (Garen should abuse passive healing)
    REWARD_PASSIVE_HEAL: float = 0.5  # Stay out of combat to heal
    REWARD_W_DAMAGE_BLOCKED: float = 1.0  # Successfully block damage with W

    # Positioning rewards
    REWARD_BUSH_USAGE: float = 0.2  # Use bushes for passive healing
    PENALTY_OVEREXTEND: float = -2.0  # Too far from tower without vision


class GarenStrategy:
    """Garen gameplay strategy for different phases"""

    EARLY_GAME = {
        "priority": "farm_and_sustain",
        "objectives": [
            "Last-hit minions safely",
            "Use Q to secure cannons",
            "Trade with Q-E when enemy wastes cooldowns",
            "Use passive to heal between waves",
            "Build towards first item (Stridebreaker or Trinity Force)"
        ],
        "avoid": [
            "Extended trades before level 6",
            "Fighting in large enemy minion waves",
            "Chasing without Q or Flash"
        ]
    }

    MID_GAME = {
        "priority": "split_push_and_pressure",
        "objectives": [
            "Push side lanes for tower plates",
            "Use E to clear waves quickly",
            "Look for R executes in skirmishes",
            "Take Herald for tower pushing",
            "Build tanky after first damage item"
        ],
        "avoid": [
            "Grouping without Teleport up",
            "Using R on tanks",
            "Fighting without W available"
        ]
    }

    LATE_GAME = {
        "priority": "frontline_and_peel",
        "objectives": [
            "Protect carries with Q silence",
            "Tank for team with W",
            "Execute low HP enemies with R",
            "Split push if team is safe",
            "Zone enemy carries with E"
        ],
        "avoid": [
            "Diving too deep without team",
            "Using R early in fight",
            "Fighting in narrow corridors (get kited)"
        ]
    }


# Garen-specific action mappings
GAREN_ACTIONS = {
    "q_auto_combo": {
        "description": "Q -> Move to enemy -> Auto attack",
        "sequence": ["q", "move_to_target", "left_click"],
        "conditions": ["q_available", "enemy_in_range_300"]
    },

    "spin_to_win": {
        "description": "E while chasing or in minion wave",
        "sequence": ["e"],
        "conditions": ["e_available", "multiple_targets_nearby"]
    },

    "execute": {
        "description": "Use R to execute low HP enemy",
        "sequence": ["r", "click_target"],
        "conditions": ["r_available", "enemy_hp_below_25_percent", "enemy_in_range_400"]
    },

    "defensive_w": {
        "description": "Use W when taking burst damage",
        "sequence": ["w"],
        "conditions": ["w_available", "enemy_about_to_damage"]
    },

    "passive_heal": {
        "description": "Back off to heal with passive",
        "sequence": ["move_to_bush", "wait"],
        "conditions": ["hp_below_50_percent", "no_combat_8_seconds"]
    }
}


# Training curriculum for Garen
GAREN_CURRICULUM = {
    "stage_1_farming": {
        "duration_steps": 100000,
        "focus": "Learn to CS with Q and E",
        "success_metric": "50+ CS per 10 minutes",
        "allowed_actions": ["move", "left_click", "q", "e"],
        "main_reward": "cs_hits"
    },

    "stage_2_trading": {
        "duration_steps": 200000,
        "focus": "Q-Auto-E combo and when to trade",
        "success_metric": "Win 60%+ of trades",
        "allowed_actions": ["move", "left_click", "q", "w", "e"],
        "main_reward": "damage_ratio"
    },

    "stage_3_all_in": {
        "duration_steps": 300000,
        "focus": "Full combo with R execute",
        "success_metric": "3+ kills with <2 deaths per game",
        "allowed_actions": ["move", "left_click", "q", "w", "e", "r", "d", "f"],
        "main_reward": "kills_and_kda"
    },

    "stage_4_macro": {
        "duration_steps": 500000,
        "focus": "Split pushing and objective control",
        "success_metric": "Take 2+ towers per game",
        "allowed_actions": ["all"],
        "main_reward": "objectives_and_gold"
    }
}


def get_garen_reward(game_state: dict, action: dict, previous_state: dict) -> float:
    """
    Calculate Garen-specific reward based on gameplay.

    This is a simplified version - full implementation would need
    game state tracking and ability cooldown detection.
    """
    reward = 0.0

    # Example: Reward for using Q before auto-attack
    if action.get("keyboard") == "q" and previous_state.get("next_action") == "auto":
        reward += GarenRewards.REWARD_Q_AUTO_COMBO

    # Example: Reward for R execute
    if action.get("keyboard") == "r" and game_state.get("enemy_hp") < 0.25:
        reward += GarenRewards.REWARD_R_EXECUTE

    # Penalize R on full HP enemy
    if action.get("keyboard") == "r" and game_state.get("enemy_hp") > 0.7:
        reward += GarenRewards.REWARD_R_WASTE

    return reward


# Export for use in training
garen_abilities = GarenAbilities()
garen_combos = GarenCombos()
garen_rewards = GarenRewards()
garen_strategy = GarenStrategy()
