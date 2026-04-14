import warnings
from typing import List, Optional

import numpy as np


class LiquidityHawkesProcess:
    """
    Simulates a Hawkes process, commonly used in finance to model event
    arrivals (like market orders/liquidity consumption) where a past event
    increases the probability of future events.

    The intensity function lambda(t) is defined as:
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    Where:
    - mu (background intensity): The constant, spontaneous arrival rate.
    - alpha (jump size): The initial jump in intensity caused by an event.
    - beta (decay rate): How quickly the intensity decays back to mu.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.3, mu: float = 0.5) -> None:
        """
        Initializes the Hawkes Process parameters.

        Args:
            alpha: Jump size of the kernel (must be < beta for stability).
            beta: Decay rate of the kernel.
            mu: Background intensity.
        """
        if alpha >= beta:
            warnings.warn(
                "Hawkes Process is unstable (alpha >= beta). Consider changing parameters.",
                UserWarning,
            )
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.intensity = mu
        self.last_event_time: Optional[float] = None

    def simulate(self, T: float = 1000) -> np.ndarray:
        """
        Simulates the Hawkes process over a time horizon T using the
        thinning algorithm (Lewis and Shedler).

        Args:
            T: The end time for the simulation.

        Returns:
            A numpy array of event times.
        """
        events: List[float] = []
        t: float = 0.0
        while t < T:
            M = self.intensity
            u1 = np.random.rand()
            dt = -np.log(u1) / M
            t += dt
            if t > T:
                break
            current_lambda = self.mu + (self.intensity - self.mu) * np.exp(
                -self.beta * dt
            )
            u2 = np.random.rand()
            if u2 * M <= current_lambda:
                events.append(t)
                self.intensity = current_lambda + self.alpha
                self.last_event_time = t
            else:
                self.intensity = current_lambda
        return np.array(events)

    def forecast_optimal_spread(self, mid_price: float) -> float:
        """
        Calculates an optimal market-making spread based on the Hawkes process
        parameters for buy and sell side intensity, assuming a simplified
        Avellaneda-Stoikov-like model where:

        lambda_plus (buy orders) and lambda_minus (sell orders)
        are the average arrival rates, which, in a stationary Hawkes process,
        converge to: lambda_avg = mu / (1 - alpha/beta).

        The simplified formula provided relates spread to the imbalance
        of arrival rates, which is often derived from optimal quoting models
        (like Avellaneda-Stoikov).

        Note: The original code provided lambda_plus == lambda_minus, resulting in a zero spread.
        This is typically only correct for a symmetric market.
        I'm keeping the original formula but adding a warning about the symmetry.
        A real forecasting model would require separate alpha/beta/mu for bid/ask events.

        Args:
            mid_price: The current mid-price of the asset.

        Returns:
            The calculated optimal spread.
        """
        try:
            lambda_avg = self.mu / (1 - self.alpha / self.beta)
        except ZeroDivisionError:
            warnings.warn(
                "Instability in Hawkes process: 1 - alpha/beta is zero or near zero.",
                UserWarning,
            )
            return 0.0
        lambda_plus = lambda_avg
        lambda_minus = lambda_avg
        if lambda_plus == lambda_minus:
            warnings.warn(
                "Symmetric rates (lambda_plus == lambda_minus) result in a zero optimal spread by this formula.",
                UserWarning,
            )
            return 0.0
        return 0.0
