"""
Reinforcement Learning Agent Core
Implements PPO with LSTM for temporal memory and hybrid action space.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .config import rl_cfg, action_cfg, vision_cfg, get_device


class LSTMActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM for temporal dependencies.
    Supports hybrid continuous-discrete action space.
    """

    def __init__(
        self,
        state_dim: int = vision_cfg.STATE_DIM,
        continuous_action_dim: int = action_cfg.ACTION_DIM_CONTINUOUS,
        discrete_action_dims: Tuple[int, int] = (
            action_cfg.ACTION_DIM_DISCRETE_MOUSE,
            action_cfg.ACTION_DIM_DISCRETE_KEYBOARD
        ),
        hidden_size: int = rl_cfg.LSTM_HIDDEN_SIZE,
        num_layers: int = rl_cfg.LSTM_NUM_LAYERS,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Actor heads (policy networks)
        # Continuous actions (mouse x, mouse y, camera x, camera y)
        self.continuous_mean = nn.Linear(hidden_size, continuous_action_dim)
        self.continuous_log_std = nn.Linear(hidden_size, continuous_action_dim)

        # Discrete actions (mouse buttons and keyboard keys)
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in discrete_action_dims
        ])

        # Critic head (value function)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.continuous_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.continuous_log_std.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sequence_length: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the network.

        Args:
            state: State tensor [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            hidden_state: Optional LSTM hidden state (h, c)
            sequence_length: Length of sequence for LSTM

        Returns:
            Tuple of (continuous_actions, discrete_actions, value, new_hidden_state)
        """
        batch_size = state.shape[0]

        # Extract features
        features = self.feature_net(state)

        # If not sequential, add sequence dimension
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature_dim]

        # Process through LSTM
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(features)
        else:
            lstm_out, new_hidden = self.lstm(features, hidden_state)

        # Take last output if sequence
        if lstm_out.shape[1] > 1:
            lstm_out = lstm_out[:, -1, :]  # [batch, hidden_size]
        else:
            lstm_out = lstm_out.squeeze(1)

        # Continuous action distribution
        continuous_mean = self.continuous_mean(lstm_out)
        continuous_log_std = self.continuous_log_std(lstm_out)
        continuous_log_std = torch.clamp(continuous_log_std, -20, 2)
        continuous_std = torch.exp(continuous_log_std)

        # Discrete action logits
        discrete_logits = [head(lstm_out) for head in self.discrete_heads]

        # Value
        value = self.value_head(lstm_out)

        return continuous_mean, continuous_std, discrete_logits, value, new_hidden

    def get_action(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the policy.

        Args:
            state: State tensor
            hidden_state: LSTM hidden state
            deterministic: If True, return mean action instead of sampling

        Returns:
            (action_dict, new_hidden_state)
        """
        continuous_mean, continuous_std, discrete_logits, value, new_hidden = self.forward(
            state, hidden_state
        )

        if deterministic:
            # Use mean for continuous actions
            continuous_action = continuous_mean

            # Use argmax for discrete actions
            discrete_actions = [torch.argmax(logits, dim=-1) for logits in discrete_logits]
        else:
            # Sample continuous actions from normal distribution
            continuous_dist = Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()

            # Sample discrete actions from categorical distribution
            discrete_actions = [
                Categorical(logits=logits).sample() for logits in discrete_logits
            ]

        action_dict = {
            "continuous": continuous_action,
            "discrete_mouse": discrete_actions[0],
            "discrete_keyboard": discrete_actions[1],
            "value": value
        }

        return action_dict, new_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,
        continuous_actions: torch.Tensor,
        discrete_actions: Tuple[torch.Tensor, ...],
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            states: State tensor
            continuous_actions: Continuous actions taken
            discrete_actions: Tuple of discrete actions taken
            hidden_state: LSTM hidden state

        Returns:
            (log_probs, entropy, values)
        """
        continuous_mean, continuous_std, discrete_logits, values, _ = self.forward(
            states, hidden_state
        )

        # Continuous action log probabilities
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_actions).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)

        # Discrete action log probabilities
        discrete_log_probs = []
        discrete_entropies = []

        for i, (logits, actions) in enumerate(zip(discrete_logits, discrete_actions)):
            dist = Categorical(logits=logits)
            discrete_log_probs.append(dist.log_prob(actions))
            discrete_entropies.append(dist.entropy())

        # Combine log probs and entropies
        total_log_prob = continuous_log_prob + sum(discrete_log_probs)
        total_entropy = continuous_entropy + sum(discrete_entropies)

        return total_log_prob, total_entropy, values.squeeze(-1)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(
        self,
        state_dim: int = vision_cfg.STATE_DIM,
        learning_rate: float = rl_cfg.PPO_LEARNING_RATE,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.state_dim = state_dim

        # Create actor-critic network
        self.policy = LSTMActorCritic(state_dim=state_dim).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Training hyperparameters
        self.gamma = rl_cfg.PPO_GAMMA
        self.gae_lambda = rl_cfg.PPO_GAE_LAMBDA
        self.clip_range = rl_cfg.PPO_CLIP_RANGE
        self.ent_coef = rl_cfg.PPO_ENT_COEF
        self.vf_coef = rl_cfg.PPO_VF_COEF
        self.max_grad_norm = rl_cfg.PPO_MAX_GRAD_NORM

        # Hidden state for LSTM
        self.hidden_state = None

        # Training stats
        self.n_updates = 0

    def reset_hidden_state(self):
        """Reset LSTM hidden state (call at episode start)"""
        self.hidden_state = None

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Select an action given the current state.

        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with action components
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get action from policy
        with torch.no_grad():
            action_dict, self.hidden_state = self.policy.get_action(
                state_tensor,
                self.hidden_state,
                deterministic=deterministic
            )

        # Convert to numpy
        result = {
            "continuous": action_dict["continuous"].cpu().numpy()[0],
            "discrete_mouse": action_dict["discrete_mouse"].cpu().item(),
            "discrete_keyboard": action_dict["discrete_keyboard"].cpu().item(),
            "value": action_dict["value"].cpu().item()
        }

        return result

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of value predictions
            dones: Array of done flags
            next_value: Value of next state

        Returns:
            (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(
        self,
        rollout_buffer: Dict[str, np.ndarray],
        n_epochs: int = rl_cfg.PPO_N_EPOCHS,
        batch_size: int = rl_cfg.PPO_BATCH_SIZE
    ) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            rollout_buffer: Dictionary containing rollout data
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary of training metrics
        """
        # Extract rollout data
        states = torch.FloatTensor(rollout_buffer["states"]).to(self.device)
        continuous_actions = torch.FloatTensor(rollout_buffer["continuous_actions"]).to(self.device)
        discrete_mouse_actions = torch.LongTensor(rollout_buffer["discrete_mouse_actions"]).to(self.device)
        discrete_keyboard_actions = torch.LongTensor(rollout_buffer["discrete_keyboard_actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_batches = 0

        # Multiple epochs over the data
        for epoch in range(n_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))

            for start_idx in range(0, len(states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                # Get batch data
                batch_states = states[batch_indices]
                batch_continuous_actions = continuous_actions[batch_indices]
                batch_discrete_mouse = discrete_mouse_actions[batch_indices]
                batch_discrete_keyboard = discrete_keyboard_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_states,
                    batch_continuous_actions,
                    (batch_discrete_mouse, batch_discrete_keyboard)
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_batches += 1

        self.n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy_loss": total_entropy_loss / n_batches,
            "n_updates": self.n_updates
        }

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "n_updates": self.n_updates,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.n_updates = checkpoint["n_updates"]
        print(f"Model loaded from {path}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing PPO Agent")
    print("=" * 60)

    # Create agent
    agent = PPOAgent()
    print(f"Agent created on device: {agent.device}")
    print(f"Policy parameters: {sum(p.numel() for p in agent.policy.parameters())}")

    # Test action selection
    dummy_state = np.random.randn(vision_cfg.STATE_DIM)
    print("\nTesting action selection...")

    agent.reset_hidden_state()
    action = agent.select_action(dummy_state)
    print(f"Action: {action}")

    # Test training step
    print("\nTesting training update...")

    # Create dummy rollout buffer
    rollout_buffer = {
        "states": np.random.randn(128, vision_cfg.STATE_DIM),
        "continuous_actions": np.random.randn(128, action_cfg.ACTION_DIM_CONTINUOUS),
        "discrete_mouse_actions": np.random.randint(0, action_cfg.ACTION_DIM_DISCRETE_MOUSE, 128),
        "discrete_keyboard_actions": np.random.randint(0, action_cfg.ACTION_DIM_DISCRETE_KEYBOARD, 128),
        "log_probs": np.random.randn(128),
        "advantages": np.random.randn(128),
        "returns": np.random.randn(128),
    }

    metrics = agent.update(rollout_buffer, n_epochs=1, batch_size=32)
    print(f"Training metrics: {metrics}")

    # Test save/load
    print("\nTesting save/load...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pt")
        agent.save(save_path)
        agent.load(save_path)

    print("\nAll tests passed!")
