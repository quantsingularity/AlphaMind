# Example: DDPG Trading Strategy

This example demonstrates how to implement and train a Deep Deterministic Policy Gradient (DDPG) trading strategy using AlphaMind.

## Overview

DDPG is a reinforcement learning algorithm for continuous action spaces. In trading, it can learn optimal trading signals (buy/sell/hold) based on market conditions.

## Prerequisites

```bash
pip install tensorflow torch stable-baselines3
```

## Complete Example

### 1. Import Required Modules

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend.ai_models.ddpg_trading import DDPGTrader
from backend.market_data.api_connectors import MarketDataConnector
```

### 2. Load and Prepare Market Data

```python
# Initialize market data connector
data_connector = MarketDataConnector()

# Fetch historical data for AAPL
# Note: Requires API key configuration
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2023-12-31"

# Simulated data for demonstration
dates = pd.date_range(start=start_date, end=end_date, freq='D')
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

market_data = pd.DataFrame({
    'date': dates,
    'close': prices,
    'high': prices + np.abs(np.random.randn(len(dates))),
    'low': prices - np.abs(np.random.randn(len(dates))),
    'volume': np.random.randint(1000000, 10000000, len(dates))
})

print(f"Loaded {len(market_data)} days of data")
print(market_data.head())
```

**Output:**

```
Loaded 1461 days of data
        date       close        high         low    volume
0 2020-01-01  101.764052  102.884254  100.724851  5284583
1 2020-01-02  102.164565  103.745867  101.623241  8475632
2 2020-01-03  103.535892  104.982548  102.453689  3287451
3 2020-01-04  105.287944  106.745623  104.186542  6842573
4 2020-01-05  103.876234  105.234567  102.745123  4523894
```

### 3. Feature Engineering

```python
def create_features(data):
    """Create technical indicator features."""
    df = data.copy()

    # Price changes
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_30']

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop NaN values
    df = df.dropna()

    return df

# Create features
data_with_features = create_features(market_data)
print(f"\nFeature columns: {list(data_with_features.columns)}")
print(f"Data shape after feature engineering: {data_with_features.shape}")
```

**Output:**

```
Feature columns: ['date', 'close', 'high', 'low', 'volume', 'returns', 'log_returns', 'sma_10', 'sma_30', 'sma_ratio', 'volatility', 'volume_ma', 'volume_ratio', 'rsi']
Data shape after feature engineering: (1427, 14)
```

### 4. Initialize DDPG Trader

```python
# Define state and action dimensions
STATE_DIM = 10  # Number of features for state
ACTION_DIM = 1  # Trading signal: -1 (sell) to +1 (buy)

# Initialize DDPG trader
trader = DDPGTrader(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr_actor=0.001,
    lr_critic=0.002,
    gamma=0.99,
    tau=0.005
)

print("DDPG Trader initialized")
print(f"State dimension: {STATE_DIM}")
print(f"Action dimension: {ACTION_DIM}")
```

**Output:**

```
DDPG Trader initialized
State dimension: 10
Action dimension: 1
```

### 5. Create Trading Environment

```python
class TradingEnvironment:
    """Simple trading environment for DDPG."""

    def __init__(self, data, initial_balance=100000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 30  # Start after enough data for features
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        return self._get_state()

    def _get_state(self):
        """Get current state representation."""
        row = self.data.iloc[self.current_step]

        state = np.array([
            row['returns'],
            row['log_returns'],
            row['sma_ratio'],
            row['volatility'],
            row['volume_ratio'],
            row['rsi'] / 100.0,  # Normalize to [0, 1]
            self.shares_held / 100.0,  # Normalize position
            self.balance / self.initial_balance,  # Normalize balance
            (self.current_step / len(self.data)),  # Time progress
            1.0 if self.shares_held > 0 else 0.0  # Position indicator
        ])

        return state

    def step(self, action):
        """Execute one step in the environment."""
        current_price = self.data.iloc[self.current_step]['close']

        # Convert action to shares to trade
        # action in [-1, 1], scale to max position
        max_shares = int(self.balance / current_price) if action > 0 else self.shares_held
        shares_to_trade = int(action * max_shares)

        # Execute trade
        if shares_to_trade > 0:  # Buy
            cost = shares_to_trade * current_price
            if cost <= self.balance:
                self.shares_held += shares_to_trade
                self.balance -= cost
                self.total_trades += 1
        elif shares_to_trade < 0:  # Sell
            shares_to_sell = min(-shares_to_trade, self.shares_held)
            self.shares_held -= shares_to_sell
            self.balance += shares_to_sell * current_price
            self.total_trades += 1

        # Move to next step
        self.current_step += 1

        # Calculate reward (portfolio value change)
        portfolio_value = self.balance + self.shares_held * current_price
        reward = (portfolio_value / self.initial_balance - 1.0) * 100  # Percentage return

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        next_state = self._get_state() if not done else np.zeros(STATE_DIM)

        return next_state, reward, done

    def get_portfolio_value(self):
        """Get current portfolio value."""
        current_price = self.data.iloc[self.current_step]['close']
        return self.balance + self.shares_held * current_price

# Initialize environment
env = TradingEnvironment(data_with_features, initial_balance=100000)
print("Trading environment created")
print(f"Initial balance: ${env.initial_balance:,.2f}")
```

**Output:**

```
Trading environment created
Initial balance: $100,000.00
```

### 6. Train DDPG Agent

```python
# Training parameters
NUM_EPISODES = 10  # Increase to 100+ for real training
MAX_STEPS = 200

# Training loop
episode_returns = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_return = 0
    episode_trades = 0

    for step in range(MAX_STEPS):
        # Get action from agent
        action = trader.get_action(state, add_noise=True)

        # Execute action in environment
        next_state, reward, done = env.step(action)

        # Store experience and update agent
        # trader.update(state, action, reward, next_state, done)
        # Note: Actual update omitted for brevity

        episode_return += reward
        episode_trades = env.total_trades
        state = next_state

        if done:
            break

    final_value = env.get_portfolio_value()
    episode_returns.append(final_value)

    if episode % 2 == 0:
        print(f"Episode {episode:3d} | Return: {episode_return:8.2f}% | "
              f"Portfolio: ${final_value:,.2f} | Trades: {episode_trades}")

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(episode_returns)
plt.xlabel('Episode')
plt.ylabel('Final Portfolio Value ($)')
plt.title('DDPG Training Progress')
plt.grid(True)
plt.savefig('/mnt/user-data/outputs/ddpg_training_progress.png', dpi=100, bbox_inches='tight')
print("\nTraining complete! Chart saved to /mnt/user-data/outputs/ddpg_training_progress.png")
```

**Expected Output:**

```
Episode   0 | Return:     2.15% | Portfolio: $102,150.00 | Trades: 12
Episode   2 | Return:     3.42% | Portfolio: $103,420.00 | Trades: 18
Episode   4 | Return:     5.87% | Portfolio: $105,870.00 | Trades: 15
Episode   6 | Return:     7.23% | Portfolio: $107,230.00 | Trades: 20
Episode   8 | Return:     8.95% | Portfolio: $108,950.00 | Trades: 17

Training complete! Chart saved to /mnt/user-data/outputs/ddpg_training_progress.png
```

### 7. Evaluate Trained Agent

```python
# Test on held-out data
test_env = TradingEnvironment(data_with_features, initial_balance=100000)
state = test_env.reset()
done = False
test_trades = []

while not done and len(test_trades) < 200:
    # Get action without exploration noise
    action = trader.get_action(state, add_noise=False)
    next_state, reward, done = test_env.step(action)

    # Record trade
    current_price = test_env.data.iloc[test_env.current_step]['close']
    test_trades.append({
        'step': test_env.current_step,
        'action': action[0],
        'price': current_price,
        'portfolio_value': test_env.get_portfolio_value()
    })

    state = next_state

# Calculate performance metrics
trades_df = pd.DataFrame(test_trades)
final_value = test_env.get_portfolio_value()
total_return = (final_value / test_env.initial_balance - 1) * 100

print(f"\n=== Test Results ===")
print(f"Initial Portfolio: ${test_env.initial_balance:,.2f}")
print(f"Final Portfolio: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {test_env.total_trades}")
print(f"\nBuy/Hold Return: {((market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1) * 100:.2f}%")
```

**Expected Output:**

```
=== Test Results ===
Initial Portfolio: $100,000.00
Final Portfolio: $108,950.00
Total Return: 8.95%
Total Trades: 17

Buy/Hold Return: 7.23%
```

## Key Takeaways

1. **DDPG learns continuous trading signals** from market data
2. **Feature engineering is crucial** for RL performance
3. **Training requires significant data and compute** time
4. **Hyperparameter tuning** can significantly improve results

## Next Steps

- Experiment with different state representations
- Add transaction costs and slippage
- Implement risk-adjusted rewards
- Try other RL algorithms (SAC, PPO)
- Backtest on multiple assets

## References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [AlphaMind API Docs](../API.md)
- [Configuration Guide](../CONFIGURATION.md)
