import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO


class PortfolioGymEnv(gym.Env):
    """
    A custom environment for portfolio management using OpenAI Gym.
    The agent learns to allocate capital (weights) to maximize a risk-adjusted return (Sharpe Ratio).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, universe, transaction_cost=0.001):
        super().__init__()
        self.universe = universe  # List of assets (e.g., ['AAPL', 'GOOG', 'CASH'])
        self.n_assets = len(universe)
        self.transaction_cost = transaction_cost

        # Action Space: A Box of size n_assets, representing the desired new weights (or rebalancing signal)
        # We use a range of [-1, 1] before normalization (weights can be long/short)
        self.action_space = spaces.Box(-1, 1, (self.n_assets,), dtype=np.float32)

        # Observation Space: A dictionary of different data types (time series, single values)
        self.observation_space = spaces.Dict(
            {
                # Prices: (n_assets, lookback_window=10)
                "prices": spaces.Box(
                    -np.inf, np.inf, (self.n_assets, 10), dtype=np.float32
                ),
                # Volumes: (n_assets,)
                "volumes": spaces.Box(0, np.inf, (self.n_assets,), dtype=np.float32),
                # Macro features: (5,)
                "macro": spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
            }
        )

        # Internal state variables
        self.current_step = 0
        # Initialize with placeholder data (e.g., 100 days of returns for simulation)
        self.returns = np.zeros((100, self.n_assets), dtype=np.float32)
        # Initial weights are zero (or you could start with equal weights)
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32)

    def step(self, action):
        """Execute one time step within the environment."""

        # 1. Calculate and normalize the new portfolio weights
        new_weights = self._normalize_weights(action)

        # 2. Calculate the penalty for rebalancing
        cost = self._transaction_cost(new_weights)

        # 3. Calculate the reward based on the new weights
        # Reward is (Performance Measure) - (Cost Penalty)
        reward = self._sharpe_ratio(new_weights) - cost

        # 4. Advance the environment state
        self.current_step += 1

        # 5. Check if the episode is done (e.g., ran out of simulation data)
        # Assuming the episode ends after 100 steps (length of self.returns)
        done = self.current_step >= len(self.returns) - 1

        # 6. Get the next observation
        observation = self._get_obs()

        # For compatibility with gym/sb3: obs, reward, terminated, truncated, info
        return (
            observation,
            reward,
            done,
            False,
            {},
        )  # Use False for truncated for simplicity

    def _sharpe_ratio(self, weights):
        """
        Calculate the portfolio's Sharpe Ratio.
        NOTE: This is a placeholder using static returns data. In a real environment,
        this would use the returns realized *after* taking the action.
        """
        # For simulation, use a slice of the returns data for calculation
        returns = np.dot(
            self.returns[self.current_step - 10 : self.current_step], weights
        )
        risk_free_rate = (
            0.02 / 252
        )  # Daily risk-free rate approximation (assuming 252 trading days)

        # Ensure standard deviation is not zero
        std = np.std(returns)
        if std == 0:
            return -1.0  # Penalize zero volatility

        return (np.mean(returns) - risk_free_rate) / std

    def _normalize_weights(self, action):
        """Normalize actions to valid portfolio weights (e.g., sum to 1)."""

        # 1. Transform actions from [-1, 1] to a less constrained range (optional, tanh is used for smooth mapping)
        weights = np.tanh(action)

        # 2. Ensure weights sum to 1 for a fully invested portfolio (long-only or long/short with gross leverage control)
        # For simplicity: Normalize weights to sum to 1 across absolute values (i.e., gross leverage is 1)
        sum_abs_weights = np.sum(np.abs(weights))
        if sum_abs_weights == 0:
            # If all actions are zero, fall back to equal weights or a defined neutral state
            return np.ones(self.n_assets) / self.n_assets

        weights = weights / sum_abs_weights
        return weights

    def _transaction_cost(self, new_weights):
        """Calculate transaction costs from rebalancing (based on turnover)."""

        # Calculate turnover: the sum of absolute weight changes
        turnover = np.sum(np.abs(new_weights - self.current_weights))

        # Apply proportional transaction cost
        cost = turnover * self.transaction_cost

        # Update current weights for the next step's cost calculation
        self.current_weights = new_weights

        return cost

    def _get_obs(self):
        """Get current market observation (placeholder)."""

        # In a real system, this would fetch actual time-series data based on self.current_step
        # For simulation, we return random data matching the observation_space structure

        # Ensure consistent seeding for reproducibility in the real implementation

        # Create dummy data matching the observation_space structure
        return {
            "prices": np.random.normal(0, 1, (self.n_assets, 10)).astype(np.float32),
            "volumes": np.abs(np.random.normal(1000, 500, (self.n_assets,))).astype(
                np.float32
            ),
            "macro": np.random.normal(0, 1, (5,)).astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        # Start with an initial neutral portfolio (e.g., all cash or zero weights)
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32)

        # Generate random returns for the entire simulation episode
        # Mean return of 0.05% with 1% volatility (per step)
        self.returns = np.random.normal(0.0005, 0.01, (100, self.n_assets)).astype(
            np.float32
        )

        observation = self._get_obs()
        info = {}

        return observation, info


# --- RL Agent Setup ---


class PPOAgent:
    """
    A Proximal Policy Optimization (PPO) agent for training in the PortfolioGymEnv.
    """

    def __init__(self, env):
        # MultiInputPolicy is required when the observation space is a dictionary (spaces.Dict)
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,  # Print training information
        )

    def train(self, timesteps=1e6):
        """Start the training process for a specified number of time steps."""
        print(f"Starting PPO training for {timesteps/1e6} million steps...")
        # The agent interacts with the environment here (Policy Optimization Loop)
        self.model.learn(total_timesteps=int(timesteps))
        print("Training complete.")


# --- Example Usage ---

if __name__ == "__main__":
    # Define a universe of assets (e.g., 4 stocks + 1 cash component)
    asset_universe = ["StockA", "StockB", "StockC", "StockD", "Cash"]

    # 1. Initialize the Environment
    env = PortfolioGymEnv(universe=asset_universe)

    # Check the action and observation space
    print(f"Action Space Shape: {env.action_space.shape}")
    print(f"Observation Space Keys: {env.observation_space.spaces.keys()}")

    # 2. Initialize and Train the Agent
    agent = PPOAgent(env)

    # Train for a small number of steps for demonstration
    agent.train(timesteps=10000)

    # 3. Test the trained agent (Run a simulated episode)
    obs, info = env.reset()
    done = False
    episode_reward = 0

    print("\n--- Running Test Episode ---")
    while not done:
        # Agent predicts the best action based on the observation
        action, _ = agent.model.predict(obs, deterministic=True)
        # Environment executes the action
        obs, reward, done, _, info = env.step(action)
        episode_reward += reward

    print(f"Episode finished after {env.current_step} steps.")
    print(f"Total Episode Reward: {episode_reward:.4f}")
