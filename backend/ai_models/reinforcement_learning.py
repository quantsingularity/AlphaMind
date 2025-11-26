# import gym
# from gym import spaces
# import numpy as np
# from stable_baselines3 import PPO


# class PortfolioGymEnv(gym.Env):
#     def __init__(self, universe, transaction_cost=0.001):
#         self.universe = universe
#         self.action_space = spaces.Box(-1, 1, (len(universe),))
#         self.observation_space = spaces.Dict(
#             {
#                 "prices": spaces.Box(-np.inf, np.inf, (len(universe), 10)),
#                 "volumes": spaces.Box(0, np.inf, (len(universe),)),
#                 "macro": spaces.Box(-np.inf, np.inf, (5,)),
#             }
#         )
#         self.returns = np.zeros((100, len(universe)))  # Initialize with placeholder
#         self.current_weights = np.zeros(len(universe))

#     def step(self, action):
#         # Calculate portfolio rebalancing
#         new_weights = self._normalize_weights(action)
#         cost = self._transaction_cost(new_weights)
#         reward = self._sharpe_ratio(new_weights) - cost
#         return self._get_obs(), reward, False, {}

#     def _sharpe_ratio(self, weights):
#         returns = np.dot(self.returns, weights)
#         return (np.mean(returns) - 0.02) / np.std(returns)

#     def _normalize_weights(self, action):
#        """Normalize actions to valid portfolio weights"""
#        # Convert actions to weights between -1 and 1
##         weights = np.tanh(action)
#
#        # Ensure weights sum to 1 for fully invested portfolio
##         weights = weights / np.sum(np.abs(weights))
##         return weights
#
##     def _transaction_cost(self, new_weights):
#        """Calculate transaction costs from rebalancing"""
#         # Calculate turnover (sum of absolute weight changes)
#         turnover = np.sum(np.abs(new_weights - self.current_weights))

#         # Apply transaction cost
#         cost = turnover * self.transaction_cost

#         # Update current weights
#         self.current_weights = new_weights

#         return cost

#     def _get_obs(self):
#        """Get current market observation"""
#        # In a real implementation, this would fetch actual market data
#        # This is a placeholder implementation
##         return {
#            "prices": np.random.normal(0, 1, (len(self.universe), 10)),
#            "volumes": np.abs(np.random.normal(1000, 500, (len(self.universe),))),
#            "macro": np.random.normal(0, 1, (5,)),
#        }
#
##     def reset(self):
#        """Reset environment to initial state"""
#         self.current_weights = np.zeros(len(self.universe))
#         # Generate random returns for simulation
#         self.returns = np.random.normal(0.0005, 0.01, (100, len(self.universe)))
#         return self._get_obs()


# class PPOAgent:
#     def __init__(self, env):
#         self.model = PPO(
#             "MultiInputPolicy",
#             env,
#             learning_rate=3e-4,
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#         )

#     def train(self, timesteps=1e6):
#         self.model.learn(total_timesteps=int(timesteps))
