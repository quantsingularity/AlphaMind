"""
Reinforcement Learning Module for AlphaMind

This module provides reinforcement learning capabilities for adaptive trading strategies,
including various RL algorithms, environments, and training utilities.
"""

from abc import ABC, abstractmethod
import asyncio
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import math
import os
import pickle
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActionSpace(Enum):
    """Types of action spaces for RL environments."""

    DISCRETE = "discrete"  # Discrete action space (e.g., buy, sell, hold)
    CONTINUOUS = "continuous"  # Continuous action space (e.g., portfolio weights)
    MIXED = "mixed"  # Mixed action space (e.g., discrete action type with continuous parameters)


class ObservationType(Enum):
    """Types of observations for RL environments."""

    NUMERIC = "numeric"  # Numeric features
    CATEGORICAL = "categorical"  # Categorical features
    IMAGE = "image"  # Image data
    TEXT = "text"  # Text data
    MIXED = "mixed"  # Mixed data types


class RLAlgorithm(Enum):
    """Types of reinforcement learning algorithms."""

    DQN = "dqn"  # Deep Q-Network
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient
    PPO = "ppo"  # Proximal Policy Optimization
    SAC = "sac"  # Soft Actor-Critic
    A2C = "a2c"  # Advantage Actor-Critic
    TD3 = "td3"  # Twin Delayed DDPG
    CUSTOM = "custom"  # Custom algorithm


class RewardFunction(Enum):
    """Types of reward functions for RL environments."""

    RETURNS = "returns"  # Portfolio returns
    SHARPE = "sharpe"  # Sharpe ratio
    SORTINO = "sortino"  # Sortino ratio
    CALMAR = "calmar"  # Calmar ratio
    PROFIT_FACTOR = "profit_factor"  # Profit factor
    CUSTOM = "custom"  # Custom reward function


class TradingAction(Enum):
    """Trading actions for discrete action spaces."""

    BUY = 0
    SELL = 1
    HOLD = 2


class TradingEnvironment(ABC):
    """Base class for trading environments."""

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        window_size: int = 30,
        max_steps: Optional[int] = None,
        commission: float = 0.001,
        reward_function: RewardFunction = RewardFunction.RETURNS,
        initial_balance: float = 10000.0,
        action_space: ActionSpace = ActionSpace.DISCRETE,
        observation_type: ObservationType = ObservationType.NUMERIC,
        random_start: bool = True,
    ):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with market data
            features: List of feature columns to use
            window_size: Size of observation window
            max_steps: Maximum number of steps per episode
            commission: Trading commission as a fraction
            reward_function: Type of reward function
            initial_balance: Initial account balance
            action_space: Type of action space
            observation_type: Type of observation
            random_start: Whether to start at a random position
        """
        self.data = data
        self.features = features
        self.window_size = window_size
        self.max_steps = max_steps or (len(data) - window_size - 1)
        self.commission = commission
        self.reward_function = reward_function
        self.initial_balance = initial_balance
        self.action_space_type = action_space
        self.observation_type = observation_type
        self.random_start = random_start

        # Validate data
        self._validate_data()

        # Set up action and observation spaces
        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        # Initialize state
        self.reset()

    def _validate_data(self):
        """Validate input data."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("Data cannot be empty")

        for feature in self.features:
            if feature not in self.data.columns:
                raise ValueError(f"Feature '{feature}' not found in data")

        # Ensure data has a price column
        if "close" not in self.data.columns and "price" not in self.data.columns:
            raise ValueError("Data must have a 'close' or 'price' column")

        # Use 'close' as price if available, otherwise use 'price'
        self.price_col = "close" if "close" in self.data.columns else "price"

    def _setup_action_space(self) -> Dict[str, Any]:
        """
        Set up action space.

        Returns:
            Action space configuration
        """
        if self.action_space_type == ActionSpace.DISCRETE:
            # Discrete actions: buy, sell, hold
            return {
                "type": "discrete",
                "n": 3,  # Number of actions
                "actions": [a.value for a in TradingAction],
            }
        elif self.action_space_type == ActionSpace.CONTINUOUS:
            # Continuous actions: portfolio weights
            return {
                "type": "continuous",
                "shape": (1,),  # Position size as a fraction of portfolio
                "low": -1.0,  # Short position
                "high": 1.0,  # Long position
            }
        elif self.action_space_type == ActionSpace.MIXED:
            # Mixed actions: discrete action type with continuous parameters
            return {
                "type": "mixed",
                "discrete": {
                    "n": 3,  # Number of discrete actions
                    "actions": [a.value for a in TradingAction],
                },
                "continuous": {
                    "shape": (1,),  # Position size as a fraction of portfolio
                    "low": 0.0,
                    "high": 1.0,
                },
            }
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space_type}")

    def _setup_observation_space(self) -> Dict[str, Any]:
        """
        Set up observation space.

        Returns:
            Observation space configuration
        """
        if self.observation_type == ObservationType.NUMERIC:
            # Numeric features
            return {
                "type": "numeric",
                "shape": (
                    self.window_size,
                    len(self.features) + 4,
                ),  # Features + position, balance, equity, pnl
                "features": self.features + ["position", "balance", "equity", "pnl"],
            }
        elif self.observation_type == ObservationType.IMAGE:
            # Image-like representation (e.g., for CNN)
            return {
                "type": "image",
                "shape": (
                    len(self.features) + 4,
                    self.window_size,
                    1,
                ),  # Features + position, balance, equity, pnl
                "features": self.features + ["position", "balance", "equity", "pnl"],
            }
        else:
            raise ValueError(f"Unsupported observation type: {self.observation_type}")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        # Reset position and balance
        self.position = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.pnl = 0.0

        # Reset step counter
        self.current_step = 0

        # Set starting point
        if self.random_start and len(self.data) > self.window_size + self.max_steps:
            self.start_idx = random.randint(
                0, len(self.data) - self.window_size - self.max_steps - 1
            )
        else:
            self.start_idx = 0

        # Initialize history
        self.history = {
            "positions": [],
            "balances": [],
            "equities": [],
            "returns": [],
            "actions": [],
            "rewards": [],
        }

        # Get initial observation
        return self._get_observation()

    def step(
        self, action: Union[int, float, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Parse action
        action_taken = self._parse_action(action)

        # Execute action
        self._execute_action(action_taken)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Update history
        self._update_history(action_taken, reward)

        # Get info
        info = self._get_info()

        return observation, reward, done, info

    def _parse_action(self, action: Union[int, float, np.ndarray]) -> Dict[str, Any]:
        """
        Parse action based on action space.

        Args:
            action: Action from agent

        Returns:
            Parsed action
        """
        if self.action_space_type == ActionSpace.DISCRETE:
            # Discrete action
            if isinstance(action, np.ndarray):
                action = action.item()

            if action == TradingAction.BUY.value:
                return {"type": "buy", "size": 1.0}
            elif action == TradingAction.SELL.value:
                return {"type": "sell", "size": 1.0}
            else:  # HOLD
                return {"type": "hold", "size": 0.0}

        elif self.action_space_type == ActionSpace.CONTINUOUS:
            # Continuous action
            if isinstance(action, np.ndarray):
                action = action.item()

            # Clip action to valid range
            action = max(min(action, 1.0), -1.0)

            if action > 0:
                return {"type": "buy", "size": action}
            elif action < 0:
                return {"type": "sell", "size": -action}
            else:
                return {"type": "hold", "size": 0.0}

        elif self.action_space_type == ActionSpace.MIXED:
            # Mixed action
            if isinstance(action, tuple):
                action_type, action_size = action
            elif isinstance(action, np.ndarray):
                action_type = int(action[0])
                action_size = float(action[1])
            else:
                raise ValueError(
                    f"Invalid action format for mixed action space: {action}"
                )

            # Clip action size to valid range
            action_size = max(min(action_size, 1.0), 0.0)

            if action_type == TradingAction.BUY.value:
                return {"type": "buy", "size": action_size}
            elif action_type == TradingAction.SELL.value:
                return {"type": "sell", "size": action_size}
            else:  # HOLD
                return {"type": "hold", "size": 0.0}

        else:
            raise ValueError(f"Unsupported action space type: {self.action_space_type}")

    def _execute_action(self, action: Dict[str, Any]):
        """
        Execute trading action.

        Args:
            action: Parsed action
        """
        # Get current price
        current_idx = self.start_idx + self.window_size + self.current_step
        current_price = self.data.iloc[current_idx][self.price_col]

        # Calculate position value
        position_value = self.position * current_price

        if action["type"] == "buy":
            # Calculate buy amount
            if self.position <= 0:
                # If no position or short position, close it first
                self.balance += position_value * (1 - self.commission)
                self.position = 0

            # Calculate new position
            buy_amount = self.balance * action["size"]
            buy_amount_with_commission = buy_amount * (1 + self.commission)

            if buy_amount_with_commission <= self.balance:
                # Update balance and position
                self.balance -= buy_amount_with_commission
                self.position += buy_amount / current_price

        elif action["type"] == "sell":
            # Calculate sell amount
            if self.position >= 0:
                # If long position, close it first
                self.balance += position_value * (1 - self.commission)
                self.position = 0

            # Calculate new position
            sell_amount = self.balance * action["size"]
            sell_size = sell_amount / current_price

            # Update balance and position
            self.position -= sell_size
            self.balance -= sell_amount * self.commission

        # Update equity
        self.equity = self.balance + (self.position * current_price)

        # Update PnL
        if self.current_step > 0:
            prev_equity = self.history["equities"][-1]
            self.pnl = self.equity - prev_equity
        else:
            self.pnl = 0.0

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation
        """
        # Get data window
        start = self.start_idx + self.current_step
        end = start + self.window_size
        data_window = self.data.iloc[start:end].copy()

        # Extract features
        features = data_window[self.features].values

        # Normalize features
        features = self._normalize_features(features)

        # Add position, balance, equity, and pnl
        if self.current_step > 0:
            positions = np.array(
                [self.history["positions"][-self.window_size :] + [self.position]]
            )
            balances = np.array(
                [self.history["balances"][-self.window_size :] + [self.balance]]
            )
            equities = np.array(
                [self.history["equities"][-self.window_size :] + [self.equity]]
            )
            pnls = np.array([self.history["returns"][-self.window_size :] + [self.pnl]])
        else:
            positions = np.zeros((1, self.window_size))
            balances = np.ones((1, self.window_size)) * self.initial_balance
            equities = np.ones((1, self.window_size)) * self.initial_balance
            pnls = np.zeros((1, self.window_size))

        # Normalize additional features
        positions = positions / (np.max(np.abs(positions)) + 1e-6)
        balances = balances / self.initial_balance
        equities = equities / self.initial_balance
        pnls = pnls / (np.max(np.abs(pnls)) + 1e-6)

        # Combine features
        if self.observation_type == ObservationType.NUMERIC:
            # For numeric observations, stack features horizontally
            observation = np.column_stack(
                [features, positions.T, balances.T, equities.T, pnls.T]
            )
        elif self.observation_type == ObservationType.IMAGE:
            # For image observations, stack features vertically
            observation = np.vstack([features.T, positions, balances, equities, pnls])

            # Add channel dimension
            observation = observation.reshape(observation.shape + (1,))

        return observation

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features.

        Args:
            features: Feature array

        Returns:
            Normalized features
        """
        # Simple min-max normalization for each feature
        for i in range(features.shape[1]):
            feature_min = np.min(features[:, i])
            feature_max = np.max(features[:, i])

            if feature_max > feature_min:
                features[:, i] = (features[:, i] - feature_min) / (
                    feature_max - feature_min
                )
            else:
                features[:, i] = 0.0

        return features

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on reward function.

        Returns:
            Reward value
        """
        if self.current_step == 0:
            return 0.0

        if self.reward_function == RewardFunction.RETURNS:
            # Simple returns
            prev_equity = self.history["equities"][-1]
            return (self.equity - prev_equity) / prev_equity

        elif self.reward_function == RewardFunction.SHARPE:
            # Sharpe ratio (approximation)
            if len(self.history["returns"]) < 2:
                return 0.0

            returns = np.array(self.history["returns"]) / np.array(
                self.history["equities"][:-1]
            )
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-6

            return mean_return / std_return

        elif self.reward_function == RewardFunction.SORTINO:
            # Sortino ratio (approximation)
            if len(self.history["returns"]) < 2:
                return 0.0

            returns = np.array(self.history["returns"]) / np.array(
                self.history["equities"][:-1]
            )
            mean_return = np.mean(returns)

            # Calculate downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                downside_std = 1e-6
            else:
                downside_std = np.std(negative_returns) + 1e-6

            return mean_return / downside_std

        elif self.reward_function == RewardFunction.CALMAR:
            # Calmar ratio (approximation)
            if len(self.history["equities"]) < 2:
                return 0.0

            returns = np.array(self.history["returns"]) / np.array(
                self.history["equities"][:-1]
            )
            mean_return = np.mean(returns)

            # Calculate maximum drawdown
            equity_curve = np.array(self.history["equities"])
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdown) + 1e-6

            return mean_return / max_drawdown

        elif self.reward_function == RewardFunction.PROFIT_FACTOR:
            # Profit factor
            if len(self.history["returns"]) < 2:
                return 0.0

            returns = np.array(self.history["returns"])
            positive_returns = np.sum(returns[returns > 0])
            negative_returns = np.abs(np.sum(returns[returns < 0])) + 1e-6

            return positive_returns / negative_returns

        else:
            # Default to returns
            prev_equity = self.history["equities"][-1]
            return (self.equity - prev_equity) / prev_equity

    def _update_history(self, action: Dict[str, Any], reward: float):
        """
        Update history with current state.

        Args:
            action: Action taken
            reward: Reward received
        """
        self.history["positions"].append(self.position)
        self.history["balances"].append(self.balance)
        self.history["equities"].append(self.equity)
        self.history["returns"].append(self.pnl)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information.

        Returns:
            Information dictionary
        """
        current_idx = self.start_idx + self.window_size + self.current_step
        current_price = self.data.iloc[current_idx][self.price_col]

        return {
            "step": self.current_step,
            "price": current_price,
            "position": self.position,
            "balance": self.balance,
            "equity": self.equity,
            "pnl": self.pnl,
            "total_return": (self.equity - self.initial_balance) / self.initial_balance,
        }

    def render(self, mode: str = "human"):
        """
        Render the environment.

        Args:
            mode: Rendering mode
        """
        if mode == "human":
            current_idx = self.start_idx + self.window_size + self.current_step
            current_price = self.data.iloc[current_idx][self.price_col]

            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Position: {self.position:.6f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Equity: {self.equity:.2f}")
            print(f"PnL: {self.pnl:.2f}")
            print(
                f"Total Return: {(self.equity - self.initial_balance) / self.initial_balance:.2%}"
            )
            print("-" * 50)

    def close(self):
        """Close environment and release resources."""
        pass


class ReplayBuffer:
    """Experience replay buffer for RL algorithms."""

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Batch of experiences
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        """
        Get buffer length.

        Returns:
            Buffer length
        """
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces."""

    def __init__(
        self, input_shape: Tuple[int, ...], num_actions: int, hidden_size: int = 128
    ):
        """
        Initialize DQN network.

        Args:
            input_shape: Shape of input observations
            num_actions: Number of discrete actions
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()

        # Determine input size
        if len(input_shape) == 2:
            # Numeric observations (window_size, features)
            input_size = input_shape[0] * input_shape[1]
            self.flatten = lambda x: x.view(x.size(0), -1)

            # Create network
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions),
            )

        elif len(input_shape) == 3:
            # Image-like observations (features, window_size, channels)
            self.flatten = lambda x: x  # No flattening needed

            # Create CNN network
            self.network = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(
                    64 * (input_shape[1] // 4) * (input_shape[2] // 4), hidden_size
                ),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions),
            )

        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Q-values for each action
        """
        x = self.flatten(x)
        return self.network(x)


class DQNAgent:
    """Deep Q-Network agent for discrete action spaces."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update: int = 10,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        device: str = "auto",
    ):
        """
        Initialize DQN agent.

        Args:
            input_shape: Shape of input observations
            num_actions: Number of discrete actions
            hidden_size: Size of hidden layers
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            target_update: Target network update frequency
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            device: Device to use (auto, cpu, cuda)
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create networks
        self.policy_net = DQNNetwork(input_shape, num_actions, hidden_size).to(
            self.device
        )
        self.target_net = DQNNetwork(input_shape, num_actions, hidden_size).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Set hyperparameters
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size

        # Initialize counters
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether agent is training

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.num_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def update(self) -> Optional[float]:
        """
        Update agent.

        Returns:
            Loss value or None if buffer is too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update counter
        self.steps_done += 1

        return loss.item()

    def train(
        self,
        env: TradingEnvironment,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
        eval_interval: int = 10,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train agent.

        Args:
            env: Trading environment
            num_episodes: Number of episodes
            max_steps_per_episode: Maximum steps per episode
            eval_interval: Evaluation interval
            render: Whether to render environment
            verbose: Whether to print progress

        Returns:
            Training metrics
        """
        # Initialize metrics
        metrics = {
            "episode_rewards": [],
            "episode_returns": [],
            "episode_lengths": [],
            "eval_returns": [],
        }

        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0

            # Set max steps
            max_steps = max_steps_per_episode or env.max_steps

            for step in range(max_steps):
                # Select action
                action = self.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Add to replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update agent
                loss = self.update()

                # Update state
                state = next_state
                episode_reward += reward

                # Render if requested
                if render:
                    env.render()

                if done:
                    break

            # Update metrics
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_returns"].append(info["total_return"])
            metrics["episode_lengths"].append(step + 1)

            # Update counter
            self.episodes_done += 1

            # Evaluate if needed
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate(env)
                metrics["eval_returns"].append(eval_return)

                if verbose:
                    print(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"Return: {info['total_return']:.2%} | "
                        f"Eval Return: {eval_return:.2%} | "
                        f"Epsilon: {self.epsilon:.2f}"
                    )
            elif verbose:
                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Return: {info['total_return']:.2%} | "
                    f"Epsilon: {self.epsilon:.2f}"
                )

        return metrics

    def evaluate(
        self, env: TradingEnvironment, num_episodes: int = 1, render: bool = False
    ) -> float:
        """
        Evaluate agent.

        Args:
            env: Trading environment
            num_episodes: Number of episodes
            render: Whether to render environment

        Returns:
            Average return
        """
        returns = []

        for _ in range(num_episodes):
            # Reset environment
            state = env.reset()
            done = False

            while not done:
                # Select action
                action = self.select_action(state, training=False)

                # Take step
                state, _, done, info = env.step(action)

                # Render if requested
                if render:
                    env.render()

            # Add return
            returns.append(info["total_return"])

        # Return average
        return sum(returns) / len(returns)

    def save(self, path: str):
        """
        Save agent.

        Args:
            path: Save path
        """
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
            },
            path,
        )

    def load(self, path: str):
        """
        Load agent.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        self.episodes_done = checkpoint["episodes_done"]


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous action spaces."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        action_dim: int,
        hidden_size: int = 128,
        action_range: float = 1.0,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            input_shape: Shape of input observations
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            action_range: Range of actions
        """
        super(ActorCritic, self).__init__()

        # Determine input size
        if len(input_shape) == 2:
            # Numeric observations (window_size, features)
            input_size = input_shape[0] * input_shape[1]
            self.flatten = lambda x: x.view(x.size(0), -1)

            # Create shared network
            self.shared = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

            # Create actor network
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim * 2),  # Mean and log_std
            )

            # Create critic network
            self.critic = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        elif len(input_shape) == 3:
            # Image-like observations (features, window_size, channels)
            self.flatten = lambda x: x  # No flattening needed

            # Create shared CNN network
            self.shared = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(
                    64 * (input_shape[1] // 4) * (input_shape[2] // 4), hidden_size
                ),
                nn.ReLU(),
            )

            # Create actor network
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim * 2),  # Mean and log_std
            )

            # Create critic network
            self.critic = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        self.action_dim = action_dim
        self.action_range = action_range

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (action_dist, value)
        """
        x = self.flatten(x)
        shared_features = self.shared(x)

        # Actor output
        actor_output = self.actor(shared_features)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)

        # Critic output
        value = self.critic(shared_features)

        return (mean, log_std), value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.

        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, log_prob, value)
        """
        (mean, log_std), value = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            action = normal.rsample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        # Scale action to range
        scaled_action = torch.tanh(action) * self.action_range

        return scaled_action, log_prob, value


class PPOAgent:
    """Proximal Policy Optimization agent for continuous action spaces."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        action_dim: int,
        hidden_size: int = 128,
        action_range: float = 1.0,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        device: str = "auto",
    ):
        """
        Initialize PPO agent.

        Args:
            input_shape: Shape of input observations
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            action_range: Range of actions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm
            update_epochs: Number of update epochs per batch
            device: Device to use (auto, cpu, cuda)
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create network
        self.network = ActorCritic(
            input_shape, action_dim, hidden_size, action_range
        ).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Set hyperparameters
        self.action_dim = action_dim
        self.action_range = action_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        # Initialize buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action using policy.

        Args:
            state: Current state
            training: Whether agent is training

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(
                state_tensor, deterministic=not training
            )

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0] if log_prob is not None else 0.0,
                value.cpu().numpy()[0],
            )

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """
        Store transition in buffer.

        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            value: Value estimate
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def update(self) -> Dict[str, float]:
        """
        Update agent.

        Returns:
            Dictionary of loss values
        """
        # Convert buffers to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0

        for _ in range(self.update_epochs):
            # Get action distribution and values
            (mean, log_std), new_values = self.network(states)
            std = log_std.exp()

            # Create normal distribution
            normal = Normal(mean, std)
            new_log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )

            # Compute actor loss
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Compute critic loss
            critic_loss = F.mse_loss(new_values, returns)

            # Compute entropy loss
            entropy_loss = -entropy.mean()

            # Compute total loss
            loss = (
                actor_loss
                + self.value_coef * critic_loss
                + self.entropy_coef * entropy_loss
            )

            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            actor_loss += actor_loss.item()
            critic_loss += critic_loss.item()
            entropy_loss += entropy_loss.item()

        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Return average losses
        return {
            "total_loss": total_loss / self.update_epochs,
            "actor_loss": actor_loss / self.update_epochs,
            "critic_loss": critic_loss / self.update_epochs,
            "entropy_loss": entropy_loss / self.update_epochs,
        }

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation.

        Args:
            rewards: Rewards tensor
            values: Values tensor
            dones: Dones tensor

        Returns:
            Tuple of (returns, advantages)
        """
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Initialize running values
        next_value = 0
        next_advantage = 0

        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # Compute return
            returns[t] = rewards[t] + self.gamma * next_value * (1 - dones[t])

            # Compute TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # Compute advantage
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (
                1 - dones[t]
            )

            # Update running values
            next_value = values[t]
            next_advantage = advantages[t]

        return returns, advantages

    def train(
        self,
        env: TradingEnvironment,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
        update_interval: int = 2048,
        eval_interval: int = 10,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train agent.

        Args:
            env: Trading environment
            num_episodes: Number of episodes
            max_steps_per_episode: Maximum steps per episode
            update_interval: Update interval
            eval_interval: Evaluation interval
            render: Whether to render environment
            verbose: Whether to print progress

        Returns:
            Training metrics
        """
        # Initialize metrics
        metrics = {
            "episode_rewards": [],
            "episode_returns": [],
            "episode_lengths": [],
            "eval_returns": [],
            "losses": [],
        }

        # Initialize counters
        total_steps = 0

        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0

            # Set max steps
            max_steps = max_steps_per_episode or env.max_steps

            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store transition
                self.store_transition(state, action, log_prob, reward, value, done)

                # Update state
                state = next_state
                episode_reward += reward
                total_steps += 1

                # Render if requested
                if render:
                    env.render()

                # Update if needed
                if total_steps % update_interval == 0:
                    losses = self.update()
                    metrics["losses"].append(losses["total_loss"])

                if done:
                    break

            # Update metrics
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_returns"].append(info["total_return"])
            metrics["episode_lengths"].append(step + 1)

            # Evaluate if needed
            if (episode + 1) % eval_interval == 0:
                eval_return = self.evaluate(env)
                metrics["eval_returns"].append(eval_return)

                if verbose:
                    print(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"Return: {info['total_return']:.2%} | "
                        f"Eval Return: {eval_return:.2%}"
                    )
            elif verbose:
                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Return: {info['total_return']:.2%}"
                )

        return metrics

    def evaluate(
        self, env: TradingEnvironment, num_episodes: int = 1, render: bool = False
    ) -> float:
        """
        Evaluate agent.

        Args:
            env: Trading environment
            num_episodes: Number of episodes
            render: Whether to render environment

        Returns:
            Average return
        """
        returns = []

        for _ in range(num_episodes):
            # Reset environment
            state = env.reset()
            done = False

            while not done:
                # Select action
                action, _, _ = self.select_action(state, training=False)

                # Take step
                state, _, done, info = env.step(action)

                # Render if requested
                if render:
                    env.render()

            # Add return
            returns.append(info["total_return"])

        # Return average
        return sum(returns) / len(returns)

    def save(self, path: str):
        """
        Save agent.

        Args:
            path: Save path
        """
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """
        Load agent.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class RLTrader:
    """Reinforcement learning trader for AlphaMind."""

    def __init__(
        self,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        observation_type: ObservationType = ObservationType.NUMERIC,
        action_space: ActionSpace = ActionSpace.DISCRETE,
        reward_function: RewardFunction = RewardFunction.RETURNS,
        window_size: int = 30,
        commission: float = 0.001,
        initial_balance: float = 10000.0,
        device: str = "auto",
    ):
        """
        Initialize RL trader.

        Args:
            algorithm: RL algorithm
            observation_type: Observation type
            action_space: Action space
            reward_function: Reward function
            window_size: Observation window size
            commission: Trading commission
            initial_balance: Initial account balance
            device: Device to use (auto, cpu, cuda)
        """
        self.algorithm = algorithm
        self.observation_type = observation_type
        self.action_space = action_space
        self.reward_function = reward_function
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        self.device = device

        self.env = None
        self.agent = None
        self.features = None
        self.is_trained = False

    def create_environment(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        random_start: bool = True,
    ) -> TradingEnvironment:
        """
        Create trading environment.

        Args:
            data: DataFrame with market data
            features: List of feature columns to use
            max_steps: Maximum number of steps per episode
            random_start: Whether to start at a random position

        Returns:
            Trading environment
        """
        # Set default features if not provided
        if features is None:
            # Use OHLCV columns if available
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [col for col in ohlcv_cols if col in data.columns]

            if len(available_cols) > 0:
                features = available_cols
            else:
                # Use all numeric columns except date/time columns
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                features = [
                    col
                    for col in numeric_cols
                    if not any(
                        time_col in col.lower()
                        for time_col in ["date", "time", "timestamp"]
                    )
                ]

        self.features = features

        # Create environment
        self.env = TradingEnvironment(
            data=data,
            features=features,
            window_size=self.window_size,
            max_steps=max_steps,
            commission=self.commission,
            reward_function=self.reward_function,
            initial_balance=self.initial_balance,
            action_space=self.action_space,
            observation_type=self.observation_type,
            random_start=random_start,
        )

        return self.env

    def create_agent(self) -> Union[DQNAgent, PPOAgent]:
        """
        Create RL agent.

        Returns:
            RL agent
        """
        if self.env is None:
            raise ValueError("Environment not created")

        # Get input shape
        if self.observation_type == ObservationType.NUMERIC:
            input_shape = (
                self.window_size,
                len(self.features) + 4,
            )  # Features + position, balance, equity, pnl
        elif self.observation_type == ObservationType.IMAGE:
            input_shape = (
                len(self.features) + 4,
                self.window_size,
                1,
            )  # Features + position, balance, equity, pnl
        else:
            raise ValueError(f"Unsupported observation type: {self.observation_type}")

        # Create agent based on algorithm
        if self.algorithm == RLAlgorithm.DQN:
            if self.action_space != ActionSpace.DISCRETE:
                raise ValueError("DQN only supports discrete action spaces")

            self.agent = DQNAgent(
                input_shape=input_shape,
                num_actions=self.env.action_space["n"],
                device=self.device,
            )

        elif self.algorithm == RLAlgorithm.PPO:
            if self.action_space == ActionSpace.DISCRETE:
                # For discrete actions, use one-hot encoding
                action_dim = self.env.action_space["n"]
            else:
                # For continuous actions, use scalar
                action_dim = 1

            self.agent = PPOAgent(
                input_shape=input_shape, action_dim=action_dim, device=self.device
            )

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return self.agent

    def train(
        self,
        data: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        num_episodes: int = 100,
        max_steps_per_episode: Optional[int] = None,
        eval_interval: int = 10,
        save_path: Optional[str] = None,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train RL trader.

        Args:
            data: DataFrame with market data
            features: List of feature columns to use
            num_episodes: Number of episodes
            max_steps_per_episode: Maximum steps per episode
            eval_interval: Evaluation interval
            save_path: Path to save trained agent
            render: Whether to render environment
            verbose: Whether to print progress

        Returns:
            Training metrics
        """
        # Create environment if not exists or if new data provided
        if self.env is None or data is not None:
            self.create_environment(data, features, max_steps_per_episode)

        # Create agent if not exists
        if self.agent is None:
            self.create_agent()

        # Train agent
        if self.algorithm == RLAlgorithm.DQN:
            metrics = self.agent.train(
                env=self.env,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode,
                eval_interval=eval_interval,
                render=render,
                verbose=verbose,
            )

        elif self.algorithm == RLAlgorithm.PPO:
            metrics = self.agent.train(
                env=self.env,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode,
                eval_interval=eval_interval,
                render=render,
                verbose=verbose,
            )

        # Save agent if requested
        if save_path is not None:
            self.save(save_path)

        self.is_trained = True

        return metrics

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> Union[int, float, np.ndarray]:
        """
        Predict action for state.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Predicted action
        """
        if self.agent is None:
            raise ValueError("Agent not created")

        if not self.is_trained:
            logger.warning("Agent not trained")

        if self.algorithm == RLAlgorithm.DQN:
            return self.agent.select_action(state, training=not deterministic)

        elif self.algorithm == RLAlgorithm.PPO:
            action, _, _ = self.agent.select_action(state, training=not deterministic)
            return action

    def backtest(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Backtest RL trader.

        Args:
            data: DataFrame with market data
            features: List of feature columns to use
            render: Whether to render environment
            verbose: Whether to print progress

        Returns:
            Backtest results
        """
        if self.agent is None:
            raise ValueError("Agent not created")

        if not self.is_trained:
            logger.warning("Agent not trained")

        # Create environment for backtesting
        backtest_env = self.create_environment(
            data=data, features=features, random_start=False
        )

        # Reset environment
        state = backtest_env.reset()
        done = False

        # Initialize results
        results = {
            "actions": [],
            "positions": [],
            "balances": [],
            "equities": [],
            "returns": [],
            "prices": [],
        }

        # Run backtest
        while not done:
            # Predict action
            action = self.predict(state, deterministic=True)

            # Take step
            state, _, done, info = backtest_env.step(action)

            # Store results
            results["actions"].append(action)
            results["positions"].append(info["position"])
            results["balances"].append(info["balance"])
            results["equities"].append(info["equity"])
            results["returns"].append(info["pnl"])
            results["prices"].append(info["price"])

            # Render if requested
            if render:
                backtest_env.render()

            # Print progress if verbose
            if verbose and backtest_env.current_step % 100 == 0:
                print(
                    f"Step {backtest_env.current_step}/{backtest_env.max_steps} | "
                    f"Equity: {info['equity']:.2f} | "
                    f"Return: {info['total_return']:.2%}"
                )

        # Calculate performance metrics
        equity_curve = np.array(results["equities"])
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # Calculate maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)

        # Calculate win rate
        wins = np.sum(np.array(results["returns"]) > 0)
        total_trades = len(results["returns"])
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Add performance metrics to results
        results["performance"] = {
            "initial_equity": results["equities"][0],
            "final_equity": results["equities"][-1],
            "total_return": (results["equities"][-1] - results["equities"][0])
            / results["equities"][0],
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

        if verbose:
            print(f"Backtest Results:")
            print(f"Initial Equity: ${results['equities'][0]:.2f}")
            print(f"Final Equity: ${results['equities'][-1]:.2f}")
            print(f"Total Return: {results['performance']['total_return']:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Win Rate: {win_rate:.2%}")

        return results

    def save(self, path: str):
        """
        Save RL trader.

        Args:
            path: Save path
        """
        if self.agent is None:
            raise ValueError("Agent not created")

        # Create directory if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save agent
        self.agent.save(path)

        # Save metadata
        metadata = {
            "algorithm": self.algorithm.value,
            "observation_type": self.observation_type.value,
            "action_space": self.action_space.value,
            "reward_function": self.reward_function.value,
            "window_size": self.window_size,
            "commission": self.commission,
            "initial_balance": self.initial_balance,
            "features": self.features,
            "is_trained": self.is_trained,
        }

        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path: str):
        """
        Load RL trader.

        Args:
            path: Load path
        """
        # Load metadata
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)

        # Set attributes
        self.algorithm = RLAlgorithm(metadata["algorithm"])
        self.observation_type = ObservationType(metadata["observation_type"])
        self.action_space = ActionSpace(metadata["action_space"])
        self.reward_function = RewardFunction(metadata["reward_function"])
        self.window_size = metadata["window_size"]
        self.commission = metadata["commission"]
        self.initial_balance = metadata["initial_balance"]
        self.features = metadata["features"]
        self.is_trained = metadata["is_trained"]

        # Create dummy environment
        dummy_data = pd.DataFrame(
            {
                "close": np.ones(self.window_size + 1),
                **{feature: np.ones(self.window_size + 1) for feature in self.features},
            }
        )

        self.create_environment(dummy_data, self.features)

        # Create agent
        self.create_agent()

        # Load agent
        self.agent.load(path)


# Example usage
def example_usage():
    """Example of how to use the reinforcement learning module."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Generate price series with trend and noise
    price = 100.0
    prices = [price]

    for _ in range(n_samples - 1):
        # Add trend and noise
        price = price * (1 + np.random.normal(0.0001, 0.01))
        prices.append(price)

    # Create DataFrame
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 - np.random.uniform(0, 0.005, n_samples)),
            "high": prices * (1 + np.random.uniform(0, 0.01, n_samples)),
            "low": prices * (1 - np.random.uniform(0, 0.01, n_samples)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        }
    )

    # Add technical indicators
    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_30"] = data["close"].rolling(30).mean()
    data["rsi"] = 50 + np.random.normal(0, 10, n_samples)  # Simplified RSI

    # Drop NaN values
    data = data.dropna()

    # Split data into train and test sets
    train_data = data.iloc[: int(len(data) * 0.8)]
    test_data = data.iloc[int(len(data) * 0.8) :]

    # Create RL trader
    trader = RLTrader(
        algorithm=RLAlgorithm.DQN,
        observation_type=ObservationType.NUMERIC,
        action_space=ActionSpace.DISCRETE,
        reward_function=RewardFunction.RETURNS,
        window_size=30,
        commission=0.001,
        initial_balance=10000.0,
    )

    # Train trader
    features = ["open", "high", "low", "close", "volume", "sma_10", "sma_30", "rsi"]
    metrics = trader.train(
        data=train_data,
        features=features,
        num_episodes=10,
        eval_interval=2,
        verbose=True,
    )

    # Backtest trader
    results = trader.backtest(data=test_data, features=features, verbose=True)

    print(f"Training metrics: {metrics}")
    print(f"Backtest results: {results['performance']}")


if __name__ == "__main__":
    example_usage()
