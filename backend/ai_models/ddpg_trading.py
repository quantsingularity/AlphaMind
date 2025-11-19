# from collections import deque, namedtuple
# from datetime import datetime
# import json
# import logging
# import os
# import random

# import gym
# from gym import spaces
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("ddpg_trading.log"), logging.StreamHandler()],
)
# logger = logging.getLogger("DDPG_Trading")

# Define experience replay memory
# Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


# class ReplayBuffer:
#    """Experience replay buffer to store and sample trading experiences"""
#
##     def __init__(self, capacity=100000):
##         self.buffer = deque(maxlen=capacity)
#
##     def add(self, state, action, reward, next_state, done):
#        """Add experience to buffer"""
#         experience = Experience(state, action, reward, next_state, done)
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#        """Sample random batch of experiences"""
##         experiences = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
#
#        # Convert to tensors
##         states = torch.FloatTensor(
##             [self._flatten_dict_state(e.state) for e in experiences]
#        )
##         actions = torch.FloatTensor([e.action for e in experiences])
##         rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(-1)
##         next_states = torch.FloatTensor(
##             [self._flatten_dict_state(e.next_state) for e in experiences]
#        )
##         dones = torch.FloatTensor([float(e.done) for e in experiences]).unsqueeze(-1)
#
##         return states, actions, rewards, next_states, dones
#
##     def _flatten_dict_state(self, state):
#        """Flatten dictionary state for neural network input"""
#         if isinstance(state, dict):
            # Flatten the dictionary state
#             prices = np.array(state["prices"]).flatten()
#             volumes = np.array(state["volumes"]).flatten()
#             macro = np.array(state["macro"]).flatten()
#             return np.concatenate([prices, volumes, macro])
#         return state

#     def __len__(self):
#         return len(self.buffer)


# class OUNoise:
#    """Ornstein-Uhlenbeck process for exploration noise"""
#
##     def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
##         self.mu = mu * np.ones(size)
##         self.theta = theta
##         self.sigma = sigma
##         self.size = size
##         self.reset()
#
##     def reset(self):
#        """Reset the internal state"""
#         self.state = np.copy(self.mu)

#     def sample(self):
#        """Update internal state and return noise sample"""
##         x = self.state
##         dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
##             self.size
#        )
##         self.state = x + dx
##         return self.state
#
#
## class Actor(nn.Module):
#    """Actor network for DDPG that determines the action policy"""

#     def __init__(self, state_dim, action_dim, hidden_dims=(256, 128), init_w=3e-3):
#         super(Actor, self).__init__()

        # Build network layers
#         self.layers = nn.ModuleList()
#         prev_dim = state_dim

        # Hidden layers
#         for hidden_dim in hidden_dims:
#             self.layers.append(nn.Linear(prev_dim, hidden_dim))
#             prev_dim = hidden_dim

        # Output layer
#         self.output_layer = nn.Linear(prev_dim, action_dim)

        # Initialize weights
#         self.output_layer.weight.data.uniform_(-init_w, init_w)
#         self.output_layer.bias.data.uniform_(-init_w, init_w)

#     def forward(self, state):
#        """Forward pass through the network"""
##         x = state
##         for layer in self.layers:
##             x = F.relu(layer(x))
#
#        # Output actions in range [-1, 1]
##         return torch.tanh(self.output_layer(x))
#
#
## class Critic(nn.Module):
#    """Critic network for DDPG that estimates Q-values"""

#     def __init__(self, state_dim, action_dim, hidden_dims=(256, 128), init_w=3e-3):
#         super(Critic, self).__init__()

        # First layer processes only the state
#         self.fc1 = nn.Linear(state_dim, hidden_dims[0])

        # Second layer processes both state features and action
#         self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])

        # Output layer produces Q-value
#         self.output_layer = nn.Linear(hidden_dims[1], 1)

        # Initialize weights
#         self.output_layer.weight.data.uniform_(-init_w, init_w)
#         self.output_layer.bias.data.uniform_(-init_w, init_w)

#     def forward(self, state, action):
#        """Forward pass through the network"""
#        # Process state through first layer
##         x = F.relu(self.fc1(state))
#
#        # Concatenate state features with action
##         x = torch.cat([x, action], dim=1)
#
#        # Process combined features
##         x = F.relu(self.fc2(x))
#
#        # Output Q-value
##         return self.output_layer(x)
#
#
## class DDPGAgent:
#    """Deep Deterministic Policy Gradient agent for trading"""

#     def __init__(self, env, config=None):
#        """Initialize the DDPG agent with environment and optional config"""
##         self.env = env
##         self.config = self._load_config(config)
#
#        # Extract dimensions from environment
##         if isinstance(env.observation_space, spaces.Dict):
#            # Calculate flattened state dimension
##             sample_obs = env.reset()
##             flat_obs = self._flatten_observation(sample_obs)
##             self.state_dim = len(flat_obs)
##         else:
##             self.state_dim = env.observation_space.shape[0]
#
##         self.action_dim = env.action_space.shape[0]
#
#        # Initialize device
##         self.device = torch.device(
#            "cuda" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu"
#        )
##         logger.info(f"Using device: {self.device}")
#
#        # Initialize actor and critic networks
##         self.actor = Actor(
##             self.state_dim,
##             self.action_dim,
##             hidden_dims=self.config["actor_hidden_dims"],
##         ).to(self.device)
##         self.critic = Critic(
##             self.state_dim,
##             self.action_dim,
##             hidden_dims=self.config["critic_hidden_dims"],
##         ).to(self.device)
#
#        # Initialize target networks
##         self.actor_target = Actor(
##             self.state_dim,
##             self.action_dim,
##             hidden_dims=self.config["actor_hidden_dims"],
##         ).to(self.device)
##         self.critic_target = Critic(
##             self.state_dim,
##             self.action_dim,
##             hidden_dims=self.config["critic_hidden_dims"],
##         ).to(self.device)
#
#        # Copy weights to target networks
##         self._hard_update(self.actor_target, self.actor)
##         self._hard_update(self.critic_target, self.critic)
#
#        # Initialize optimizers
##         self.actor_optimizer = optim.Adam(
##             self.actor.parameters(), lr=self.config["actor_lr"]
#        )
##         self.critic_optimizer = optim.Adam(
##             self.critic.parameters(), lr=self.config["critic_lr"]
#        )
#
#        # Initialize replay buffer
##         self.replay_buffer = ReplayBuffer(capacity=self.config["buffer_capacity"])
#
#        # Initialize exploration noise
##         self.noise = OUNoise(self.action_dim, sigma=self.config["noise_sigma"])
#
#        # Initialize training metrics
##         self.rewards_history = []
##         self.q_values_history = []
##         self.actor_losses = []
##         self.critic_losses = []
#
#        # Create results directory
##         self.results_dir = os.path.join(
#            "results", f"ddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#        )
##         os.makedirs(self.results_dir, exist_ok=True)
#
##         logger.info(
##             f"DDPG Agent initialized with state_dim={self.state_dim}, action_dim={self.action_dim}"
#        )
#
##     def _load_config(self, config=None):
#        """Load configuration with defaults"""
#         default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "actor_hidden_dims": (256, 128),
            "critic_hidden_dims": (256, 128),
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 64,
            "buffer_capacity": 100000,
            "noise_sigma": 0.2,
            "use_cuda": True,
            "log_interval": 100,
            "save_interval": 1000,
            "eval_interval": 1000,
            "max_episodes": 10000,
            "max_steps_per_episode": 1000,
            "warmup_steps": 1000,
        }

#         if config:
            # Update default config with provided values
#             default_config.update(config)

#         return default_config

#     def _flatten_observation(self, obs):
#        """Flatten dictionary observation for neural network input"""
##         if isinstance(obs, dict):
#            # Flatten the dictionary observation
##             prices = np.array(obs["prices"]).flatten()
##             volumes = np.array(obs["volumes"]).flatten()
##             macro = np.array(obs["macro"]).flatten()
##             return np.concatenate([prices, volumes, macro])
##         return obs
#
##     def _hard_update(self, target, source):
#        """Hard update: target = source"""
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(param.data)

#     def _soft_update(self, target, source):
#        """Soft update: target = tau * source + (1 - tau) * target"""
##         tau = self.config["tau"]
##         for target_param, param in zip(target.parameters(), source.parameters()):
##             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
#
##     def select_action(self, state, add_noise=True):
#        """Select action based on current policy with optional exploration noise"""
        # Flatten state if needed
#         flat_state = self._flatten_observation(state)
#         state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)

        # Set actor to evaluation mode
#         self.actor.eval()

#         with torch.no_grad():
#             action = self.actor(state_tensor).cpu().data.numpy().flatten()

        # Set actor back to training mode
#         self.actor.train()

        # Add exploration noise if required
#         if add_noise:
#             action += self.noise.sample()

        # Clip action to valid range
#         return np.clip(action, -1.0, 1.0)

#     def update(self):
#        """Update actor and critic networks using sampled experiences"""
##         if len(self.replay_buffer) < self.config["batch_size"]:
##             return 0, 0  # Not enough samples
#
#        # Sample batch from replay buffer
##         states, actions, rewards, next_states, dones = self.replay_buffer.sample(
##             self.config["batch_size"]
#        )
##         states = states.to(self.device)
##         actions = actions.to(self.device)
##         rewards = rewards.to(self.device)
##         next_states = next_states.to(self.device)
##         dones = dones.to(self.device)
#
#        # Update critic
##         with torch.no_grad():
##             next_actions = self.actor_target(next_states)
##             next_q_values = self.critic_target(next_states, next_actions)
##             target_q = rewards + (1 - dones) * self.config["gamma"] * next_q_values
#
#        # Current Q-values
##         current_q = self.critic(states, actions)
#
#        # Compute critic loss
##         critic_loss = F.mse_loss(current_q, target_q)
#
#        # Optimize critic
##         self.critic_optimizer.zero_grad()
##         critic_loss.backward()
##         self.critic_optimizer.step()
#
#        # Update actor
##         policy_actions = self.actor(states)
##         actor_loss = -self.critic(states, policy_actions).mean()
#
#        # Optimize actor
##         self.actor_optimizer.zero_grad()
##         actor_loss.backward()
##         self.actor_optimizer.step()
#
#        # Update target networks
##         self._soft_update(self.actor_target, self.actor)
##         self._soft_update(self.critic_target, self.critic)
#
#        # Record metrics
##         self.q_values_history.append(current_q.mean().item())
##         self.actor_losses.append(actor_loss.item())
##         self.critic_losses.append(critic_loss.item())
#
##         return actor_loss.item(), critic_loss.item()
#
##     def train(self, max_episodes=None, max_steps=None):
#        """Train the agent on the environment"""
#         max_episodes = max_episodes or self.config["max_episodes"]
#         max_steps = max_steps or self.config["max_steps_per_episode"]

#         logger.info(
#             f"Starting training for {max_episodes} episodes, max {max_steps} steps per episode"
        )

#         total_steps = 0
#         episode_rewards = []

#         for episode in range(1, max_episodes + 1):
#             state = self.env.reset()
#             self.noise.reset()
#             episode_reward = 0

#             for step in range(1, max_steps + 1):
                # Select action
#                 if total_steps < self.config["warmup_steps"]:
                    # Random actions during warmup
#                     action = self.env.action_space.sample()
#                 else:
                    # Policy actions with noise after warmup
#                     action = self.select_action(state, add_noise=True)

                # Execute action
#                 next_state, reward, done, info = self.env.step(action)

                # Store experience
#                 self.replay_buffer.add(state, action, reward, next_state, done)

                # Update networks
#                 if total_steps >= self.config["warmup_steps"]:
#                     actor_loss, critic_loss = self.update()

                # Update state and counters
#                 state = next_state
#                 episode_reward += reward
#                 total_steps += 1

#                 if done or step == max_steps:
#                     break

            # Record episode reward
#             episode_rewards.append(episode_reward)
#             self.rewards_history.append(episode_reward)

            # Logging
#             if episode % self.config["log_interval"] == 0:
#                 avg_reward = np.mean(episode_rewards[-self.config["log_interval"] :])
#                 logger.info(
#                     f"Episode {episode}/{max_episodes} | Avg Reward: {avg_reward:.2f} | Buffer Size: {len(self.replay_buffer)}"
                )

                # Plot and save metrics
#                 if len(self.rewards_history) > 0:
#                     self._plot_metrics()

            # Save model
#             if episode % self.config["save_interval"] == 0:
#                 self.save_model(os.path.join(self.results_dir, f"model_ep{episode}"))

            # Evaluation
#             if episode % self.config["eval_interval"] == 0:
#                 eval_reward = self.evaluate(5)
#                 logger.info(
#                     f"Evaluation after episode {episode}: Avg Reward = {eval_reward:.2f}"
                )

        # Final save
#         self.save_model(os.path.join(self.results_dir, "model_final"))

        # Final evaluation
#         final_eval_reward = self.evaluate(10)
#         logger.info(f"Final evaluation: Avg Reward = {final_eval_reward:.2f}")

        # Save final metrics
#         self._plot_metrics(final=True)

#         return self.rewards_history

#     def evaluate(self, num_episodes=5):
#        """Evaluate the agent without exploration noise"""
##         logger.info(f"Evaluating agent for {num_episodes} episodes")
##         eval_rewards = []
#
##         for episode in range(num_episodes):
##             state = self.env.reset()
##             episode_reward = 0
##             done = False
#
##             while not done:
##                 action = self.select_action(state, add_noise=False)
##                 next_state, reward, done, _ = self.env.step(action)
##                 episode_reward += reward
##                 state = next_state
#
##             eval_rewards.append(episode_reward)
#
##         avg_reward = np.mean(eval_rewards)
##         return avg_reward
#
##     def _plot_metrics(self, final=False):
#        """Plot and save training metrics"""
#         plt.figure(figsize=(15, 10))

        # Plot rewards
#         plt.subplot(2, 2, 1)
#         plt.plot(self.rewards_history)
#         plt.title("Episode Rewards")
#         plt.xlabel("Episode")
#         plt.ylabel("Reward")

        # Plot Q-values
#         if self.q_values_history:
#             plt.subplot(2, 2, 2)
#             plt.plot(self.q_values_history)
#             plt.title("Average Q-values")
#             plt.xlabel("Update Step")
#             plt.ylabel("Q-value")

        # Plot actor loss
#         if self.actor_losses:
#             plt.subplot(2, 2, 3)
#             plt.plot(self.actor_losses)
#             plt.title("Actor Loss")
#             plt.xlabel("Update Step")
#             plt.ylabel("Loss")

        # Plot critic loss
#         if self.critic_losses:
#             plt.subplot(2, 2, 4)
#             plt.plot(self.critic_losses)
#             plt.title("Critic Loss")
#             plt.xlabel("Update Step")
#             plt.ylabel("Loss")

#         plt.tight_layout()

        # Save figure
#         filename = "final_metrics.png" if final else "metrics.png"
#         plt.savefig(os.path.join(self.results_dir, filename))
#         plt.close()

#     def save_model(self, path):
#        """Save model weights and configuration"""
##         os.makedirs(path, exist_ok=True)
#
#        # Save model weights
##         torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
##         torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
##         torch.save(
##             self.actor_target.state_dict(), os.path.join(path, "actor_target.pth")
#        )
##         torch.save(
##             self.critic_target.state_dict(), os.path.join(path, "critic_target.pth")
#        )
#
#        # Save configuration
##         with open(os.path.join(path, "config.json"), "w") as f:
##             json.dump(self.config, f, indent=4)
#
##         logger.info(f"Model saved to {path}")
#
##     def load_model(self, path):
#        """Load model weights and configuration"""
        # Load model weights
#         self.actor.load_state_dict(
#             torch.load(os.path.join(path, "actor.pth"), map_location=self.device)
        )
#         self.critic.load_state_dict(
#             torch.load(os.path.join(path, "critic.pth"), map_location=self.device)
        )
#         self.actor_target.load_state_dict(
#             torch.load(os.path.join(path, "actor_target.pth"), map_location=self.device)
        )
#         self.critic_target.load_state_dict(
#             torch.load(
#                 os.path.join(path, "critic_target.pth"), map_location=self.device
            )
        )

        # Load configuration if exists
#         config_path = os.path.join(path, "config.json")
#         if os.path.exists(config_path):
#             with open(config_path, "r") as f:
#                 loaded_config = json.load(f)
#                 self.config.update(loaded_config)

#         logger.info(f"Model loaded from {path}")


# class TradingGymEnv(gym.Env):
#    """Custom Gym environment for trading with market data streams"""
#
##     def __init__(
##         self,
##         data_stream,
##         features=None,
##         window_size=10,
##         transaction_cost=0.001,
##         reward_scaling=1.0,
##         max_steps=252,
#    ):
#        """"""
#         Initialize the trading environment

#         Args:
#             data_stream: Market data stream (DataFrame or generator)
#             features: List of feature columns to use (if None, use all)
#             window_size: Number of past observations to include in state
#             transaction_cost: Cost of trading as a fraction of trade value
#             reward_scaling: Scaling factor for rewards
#             max_steps: Maximum number of steps per episode
#        """"""
##         super(TradingGymEnv, self).__init__()
#
##         self.data = data_stream
##         self.features = features
##         self.window_size = window_size
##         self.transaction_cost = transaction_cost
##         self.reward_scaling = reward_scaling
##         self.max_steps = max_steps
#
#        # Determine number of assets from data
##         if hasattr(data_stream, "shape"):
#            # DataFrame-like
##             self.num_assets = data_stream.shape[1] if len(data_stream.shape) > 1 else 1
##         else:
#            # Assume single asset if can't determine
##             self.num_assets = 1
#
#        # Define action and observation spaces
##         self.action_space = spaces.Box(
##             low=-1, high=1, shape=(self.num_assets,), dtype=np.float32
#        )
#
#        # Observation space includes price history, volumes, and macro indicators
##         self.observation_space = spaces.Dict(
#            {
#                "prices": spaces.Box(
##                     low=-np.inf,
##                     high=np.inf,
##                     shape=(self.num_assets, window_size),
##                     dtype=np.float32,
#                ),
#                "volumes": spaces.Box(
##                     low=0, high=np.inf, shape=(self.num_assets,), dtype=np.float32
#                ),
#                "macro": spaces.Box(
##                     low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
#                ),
#            }
#        )
#
#        # Initialize state variables
##         self.current_step = 0
##         self.current_weights = np.zeros(self.num_assets)
##         self.price_history = np.zeros((self.num_assets, window_size))
##         self.returns = np.zeros((100, self.num_assets))
#
#        # Logging
##         self.episode_returns = []
##         self.portfolio_values = []
##         self.actions_history = []
#
##     def reset(self):
#        """Reset environment to initial state"""
#         self.current_step = 0
#         self.current_weights = np.zeros(self.num_assets)
#         self.price_history = np.zeros((self.num_assets, self.window_size))

        # Generate random returns for simulation
        # In a real implementation, this would be replaced with actual market data
#         self.returns = np.random.normal(0.0005, 0.01, (100, self.num_assets))

        # Reset tracking variables
#         self.episode_returns = []
#         self.portfolio_values = [1.0]  # Start with $1
#         self.actions_history = []

#         return self._get_observation()

#     def step(self, action):
#        """Execute one step in the environment"""
#        # Increment step counter
##         self.current_step += 1
#
#        # Convert action to portfolio weights
##         new_weights = self._normalize_weights(action)
#
#        # Calculate transaction costs
##         cost = self._calculate_transaction_cost(new_weights)
#
#        # Calculate returns for this step
##         if self.current_step < len(self.returns):
##             step_returns = self.returns[self.current_step]
##         else:
#            # If we run out of data, generate random returns
##             step_returns = np.random.normal(0.0005, 0.01, self.num_assets)
#
#        # Calculate portfolio return
##         portfolio_return = np.sum(new_weights * step_returns) - cost
#
#        # Update portfolio value
##         current_value = self.portfolio_values[-1]
##         new_value = current_value * (1 + portfolio_return)
##         self.portfolio_values.append(new_value)
#
#        # Calculate reward (Sharpe ratio or returns-based)
##         reward = self._calculate_reward(portfolio_return)
#
#        # Update state
##         self.current_weights = new_weights
##         self.episode_returns.append(portfolio_return)
##         self.actions_history.append(new_weights)
#
#        # Update price history with new data
#        # In a real implementation, this would use actual market data
##         new_prices = self.price_history[:, 1:].copy()
##         new_prices = np.column_stack([new_prices, step_returns + 1])  # Add new prices
##         self.price_history = new_prices
#
#        # Check if episode is done
##         done = self.current_step >= self.max_steps
#
#        # Get new observation
##         observation = self._get_observation()
#
#        # Additional info
##         info = {
#            "portfolio_value": new_value,
#            "portfolio_return": portfolio_return,
#            "transaction_cost": cost,
#            "weights": new_weights,
#        }
#
##         return observation, reward, done, info
#
##     def _normalize_weights(self, action):
#        """Normalize actions to valid portfolio weights"""
        # Convert actions to weights between -1 and 1
#         weights = np.tanh(action)

        # Ensure weights sum to 1 for fully invested portfolio
#         if np.sum(np.abs(weights)) > 0:
#             weights = weights / np.sum(np.abs(weights))

#         return weights

#     def _calculate_transaction_cost(self, new_weights):
#        """Calculate transaction costs from rebalancing"""
#        # Calculate turnover (sum of absolute weight changes)
##         turnover = np.sum(np.abs(new_weights - self.current_weights))
#
#        # Apply transaction cost
##         cost = turnover * self.transaction_cost
#
##         return cost
#
##     def _calculate_reward(self, portfolio_return):
#        """Calculate reward based on portfolio performance"""
        # Simple return-based reward
#         reward = portfolio_return * self.reward_scaling

        # Alternative: Sharpe ratio if we have enough history
#         if len(self.episode_returns) > 10:
#             returns_array = np.array(self.episode_returns)
#             sharpe = (np.mean(returns_array) - 0.0001) / (np.std(returns_array) + 1e-6)
#             reward = sharpe * self.reward_scaling

#         return reward

#     def _get_observation(self):
#        """Get current market observation"""
#        # In a real implementation, this would fetch actual market data
#        # This is a placeholder implementation
##         return {
#            "prices": self.price_history,
#            "volumes": np.abs(np.random.normal(1000, 500, (self.num_assets,))),
#            "macro": np.random.normal(0, 1, (5,)),
#        }
#
##     def render(self, mode="human"):
#        """Render the environment"""
#         if mode == "human":
#             print(f"Step: {self.current_step}")
#             print(f"Portfolio Value: {self.portfolio_values[-1]:.4f}")
#             print(f"Current Weights: {self.current_weights}")
#             print(
#                 f"Current Return: {self.episode_returns[-1] if self.episode_returns else 0:.4f}"
            )
#             print("-" * 50)

#         return None


# class BacktestEngine:
#    """Backtesting engine for evaluating trading strategies"""
#
##     def __init__(self, env, agent, data=None):
#        """"""
#         Initialize backtesting engine

#         Args:
#             env: Trading environment
#             agent: Trading agent (e.g., DDPGAgent)
#             data: Optional data to use for backtesting (overrides env data)
#        """"""
##         self.env = env
##         self.agent = agent
#
##         if data is not None:
#            # Override environment data if provided
##             self.env.data = data
#
#        # Initialize metrics
##         self.portfolio_values = []
##         self.returns = []
##         self.sharpe_ratio = None
##         self.max_drawdown = None
##         self.total_return = None
##         self.annual_return = None
##         self.volatility = None
#
#        # Logging
##         self.logger = logging.getLogger("BacktestEngine")
#
##     def run(self, episodes=1, render=False):
#        """Run backtest for specified number of episodes"""
#         self.logger.info(f"Starting backtest for {episodes} episodes")

#         all_portfolio_values = []
#         all_returns = []
#         all_actions = []

#         for episode in range(1, episodes + 1):
#             state = self.env.reset()
#             done = False
#             episode_values = [1.0]  # Start with $1
#             episode_returns = []
#             episode_actions = []

#             while not done:
                # Select action without exploration noise
#                 action = self.agent.select_action(state, add_noise=False)

                # Execute action
#                 next_state, reward, done, info = self.env.step(action)

                # Record metrics
#                 episode_values.append(info["portfolio_value"])
#                 episode_returns.append(info["portfolio_return"])
#                 episode_actions.append(info["weights"])

                # Render if requested
#                 if render:
#                     self.env.render()

                # Update state
#                 state = next_state

            # Store episode results
#             all_portfolio_values.append(episode_values)
#             all_returns.append(episode_returns)
#             all_actions.append(episode_actions)

            # Log episode results
#             final_value = episode_values[-1]
#             episode_sharpe = self._calculate_sharpe_ratio(episode_returns)
#             self.logger.info(
#                 f"Episode {episode}/{episodes} | Final Value: ${final_value:.2f} | Sharpe: {episode_sharpe:.2f}"
            )

        # Calculate aggregate metrics
#         self._calculate_metrics(all_portfolio_values, all_returns, all_actions)

        # Log final results
#         self.logger.info(
#             f"Backtest completed | Total Return: {self.total_return:.2%} | Sharpe: {self.sharpe_ratio:.2f} | Max Drawdown: {self.max_drawdown:.2%}"
        )

        # Plot results
#         self._plot_results(all_portfolio_values, all_returns, all_actions)

#         return {
            "portfolio_values": all_portfolio_values,
            "returns": all_returns,
            "actions": all_actions,
            "metrics": {
                "total_return": self.total_return,
                "annual_return": self.annual_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
            },
        }

#     def _calculate_metrics(self, portfolio_values, returns, actions):
#        """Calculate performance metrics"""
#        # Use the longest episode for metrics
##         longest_idx = np.argmax([len(pv) for pv in portfolio_values])
##         self.portfolio_values = portfolio_values[longest_idx]
##         self.returns = returns[longest_idx]
#
#        # Calculate total return
##         self.total_return = self.portfolio_values[-1] / self.portfolio_values[0] - 1
#
#        # Calculate annualized return (assuming 252 trading days)
##         days = len(self.returns)
##         self.annual_return = (1 + self.total_return) ** (252 / days) - 1
#
#        # Calculate Sharpe ratio
##         self.sharpe_ratio = self._calculate_sharpe_ratio(self.returns)
#
#        # Calculate volatility
##         self.volatility = np.std(self.returns) * np.sqrt(252)
#
#        # Calculate maximum drawdown
##         self.max_drawdown = self._calculate_max_drawdown(self.portfolio_values)
#
##     def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
#        """Calculate Sharpe ratio"""
#         if not returns or len(returns) < 2:
#             return 0.0

#         returns_array = np.array(returns)
#         excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate

#         if np.std(excess_returns) == 0:
#             return 0.0

#         sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
#         return sharpe

#     def _calculate_max_drawdown(self, portfolio_values):
#        """Calculate maximum drawdown"""
##         if not portfolio_values or len(portfolio_values) < 2:
##             return 0.0
#
#        # Convert to numpy array
##         values = np.array(portfolio_values)
#
#        # Calculate running maximum
##         running_max = np.maximum.accumulate(values)
#
#        # Calculate drawdowns
##         drawdowns = (running_max - values) / running_max
#
#        # Find maximum drawdown
##         max_drawdown = np.max(drawdowns)
#
##         return max_drawdown
#
##     def _plot_results(self, portfolio_values, returns, actions):
#        """Plot backtest results"""
        # Use the longest episode for plotting
#         longest_idx = np.argmax([len(pv) for pv in portfolio_values])
#         values = portfolio_values[longest_idx]
#         rets = returns[longest_idx]
#         acts = actions[longest_idx]

        # Create figure with subplots
#         fig, axs = plt.subplots(3, 1, figsize=(12, 15))

        # Plot portfolio value
#         axs[0].plot(values)
#         axs[0].set_title("Portfolio Value")
#         axs[0].set_xlabel("Trading Day")
#         axs[0].set_ylabel("Value ($)")
#         axs[0].grid(True)

        # Plot returns
#         axs[1].plot(rets)
#         axs[1].set_title("Daily Returns")
#         axs[1].set_xlabel("Trading Day")
#         axs[1].set_ylabel("Return")
#         axs[1].grid(True)

        # Plot asset weights over time
#         acts_array = np.array(acts)
#         for i in range(acts_array.shape[1]):
#             axs[2].plot(acts_array[:, i], label=f"Asset {i+1}")

#         axs[2].set_title("Asset Weights")
#         axs[2].set_xlabel("Trading Day")
#         axs[2].set_ylabel("Weight")
#         axs[2].legend()
#         axs[2].grid(True)

        # Add metrics as text
#         metrics_text = (
#             f"Total Return: {self.total_return:.2%}\n"
#             f"Annual Return: {self.annual_return:.2%}\n"
#             f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
#             f"Volatility: {self.volatility:.2%}\n"
#             f"Max Drawdown: {self.max_drawdown:.2%}"
        )

#         fig.text(
#             0.15,
#             0.01,
#             metrics_text,
#             fontsize=12,
#             bbox=dict(facecolor="white", alpha=0.8),
        )

        # Adjust layout and save
#         plt.tight_layout()
#         plt.subplots_adjust(bottom=0.15)
#         plt.savefig("backtest_results.png")
#         plt.close()


# class HyperparameterTuner:
#    """Hyperparameter tuning for DDPG trading agent"""
#
##     def __init__(
##         self, env_creator, param_grid, n_trials=10, episodes_per_trial=5, max_steps=100
#    ):
#        """"""
#         Initialize hyperparameter tuner

#         Args:
#             env_creator: Function that creates a new environment instance
#             param_grid: Dictionary of parameters to tune with lists of values
#             n_trials: Number of random trials to run
#             episodes_per_trial: Number of episodes to run per trial
#             max_steps: Maximum steps per episode
#        """"""
##         self.env_creator = env_creator
##         self.param_grid = param_grid
##         self.n_trials = n_trials
##         self.episodes_per_trial = episodes_per_trial
##         self.max_steps = max_steps
#
#        # Initialize results storage
##         self.results = []
#
#        # Logging
##         self.logger = logging.getLogger("HyperparamTuner")
#
#        # Create results directory
##         self.results_dir = os.path.join(
#            "results", f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#        )
##         os.makedirs(self.results_dir, exist_ok=True)
#
##     def run(self):
#        """Run hyperparameter tuning"""
#         self.logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials")

#         for trial in range(1, self.n_trials + 1):
            # Sample random configuration
#             config = self._sample_config()

#             self.logger.info(f"Trial {trial}/{self.n_trials} | Config: {config}")

            # Create environment and agent
#             env = self.env_creator()
#             agent = DDPGAgent(env, config=config)

            # Train agent
#             rewards = agent.train(
#                 max_episodes=self.episodes_per_trial, max_steps=self.max_steps
            )

            # Evaluate agent
#             eval_reward = agent.evaluate(num_episodes=3)

            # Record results
#             result = {
                "trial": trial,
                "config": config,
                "eval_reward": eval_reward,
                "training_rewards": rewards,
            }

#             self.results.append(result)

#             self.logger.info(
#                 f"Trial {trial} completed | Eval Reward: {eval_reward:.2f}"
            )

        # Find best configuration
#         best_idx = np.argmax([r["eval_reward"] for r in self.results])
#         best_config = self.results[best_idx]["config"]
#         best_reward = self.results[best_idx]["eval_reward"]

#         self.logger.info(
#             f"Tuning completed | Best config: {best_config} | Reward: {best_reward:.2f}"
        )

        # Save results
#         self._save_results()

#         return best_config, self.results

#     def _sample_config(self):
#        """Sample random configuration from parameter grid"""
##         config = {}
##         for param, values in self.param_grid.items():
##             config[param] = random.choice(values)
##         return config
#
##     def _save_results(self):
#        """Save tuning results"""
        # Save all results
#         with open(os.path.join(self.results_dir, "tuning_results.json"), "w") as f:
#             json.dump(self.results, f, indent=4)

        # Plot results
#         self._plot_results()

#     def _plot_results(self):
#        """Plot tuning results"""
#        # Extract data
##         trials = [r["trial"] for r in self.results]
##         rewards = [r["eval_reward"] for r in self.results]
#
#        # Create figure
##         plt.figure(figsize=(10, 6))
##         plt.plot(trials, rewards, "o-")
##         plt.title("Hyperparameter Tuning Results")
##         plt.xlabel("Trial")
##         plt.ylabel("Evaluation Reward")
##         plt.grid(True)
#
#        # Add best trial
##         best_idx = np.argmax(rewards)
##         plt.scatter(
##             [trials[best_idx]], [rewards[best_idx]], color="red", s=100, label="Best"
#        )
##         plt.legend()
#
#        # Save figure
##         plt.savefig(os.path.join(self.results_dir, "tuning_results.png"))
##         plt.close()
#
#
## Example usage
## if __name__ == "__main__":
#    # Create environment
##     env = TradingGymEnv(
##         data_stream=None,  # Use random data for testing
##         window_size=10,
##         transaction_cost=0.001,
##         reward_scaling=1.0,
##         max_steps=252,  # One trading year
#    )
#
#    # Create agent
##     agent = DDPGAgent(env)
#
#    # Train agent
##     agent.train(max_episodes=100, max_steps=252)
#
#    # Backtest
##     backtest = BacktestEngine(env, agent)
##     results = backtest.run(episodes=1, render=True)
#
#    # Hyperparameter tuning example
##     param_grid = {
#        "actor_lr": [1e-4, 3e-4, 1e-3],
#        "critic_lr": [1e-3, 3e-3, 1e-2],
#        "gamma": [0.95, 0.97, 0.99],
#        "tau": [0.001, 0.005, 0.01],
#        "noise_sigma": [0.1, 0.2, 0.3],
#    }
#
##     tuner = HyperparameterTuner(
##         env_creator=lambda: TradingGymEnv(
##             data_stream=None, window_size=10, transaction_cost=0.001, max_steps=100
#        ),
##         param_grid=param_grid,
##         n_trials=5,
##         episodes_per_trial=3,
##         max_steps=100,
#    )
#
##     best_config, tuning_results = tuner.run()
##     print(f"Best configuration: {best_config}")
