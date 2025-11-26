import numpy as np
import warnings
from typing import List, Optional


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

    def __init__(self, alpha: float = 0.1, beta: float = 0.3, mu: float = 0.5):
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
        # Initial intensity starts at mu
        self.intensity = mu
        self.last_event_time: Optional[float] = (
            None  # Stores the time of the last event
        )

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

        # M is the majorant/upper bound for the current intensity.
        # Since intensity is non-decreasing between events, we use the current intensity.
        # Note: In a proper thinning algorithm, M should be a constant upper bound or
        # dynamically updated. Here, we use the current intensity as a simplified M,
        # which is common in self-exciting process simulations.

        # We need to maintain the full history's impact on intensity,
        # but the provided logic simplifies this by only using the *current* intensity.
        # Let's adjust the update logic to be correct for the simple self-excitation model.

        while t < T:
            # 1. Determine Majorant (M) and Time Step (dt)
            # The maximum intensity is M = lambda(t_last) + alpha.
            # We use the current self.intensity as M for the first step, then update.
            M = self.intensity

            # Draw time to next event from an exponential distribution with rate M
            u1 = np.random.rand()
            dt = -np.log(u1) / M
            t += dt

            if t > T:
                break

            # 2. Re-evaluate Intensity at t
            # The current intensity at time t is:
            # lambda(t) = mu + (intensity_at_last_event - mu) * exp(-beta * dt)
            current_lambda = self.mu + (self.intensity - self.mu) * np.exp(
                -self.beta * dt
            )

            # 3. Acceptance/Rejection Test
            u2 = np.random.rand()
            if u2 * M <= current_lambda:
                # Accept event
                events.append(t)

                # 4. Update Intensity: The new event causes a jump of alpha
                self.intensity = current_lambda + self.alpha
                self.last_event_time = t
            else:
                # Reject event (thinning)
                # The intensity should be reset to the value at time t (without the jump)
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
        # Note: The original implementation assumes lambda_plus = lambda_minus,
        # which results in a zero spread, only suitable for a perfectly
        # symmetric market.

        try:
            lambda_avg = self.mu / (1 - self.alpha / self.beta)
        except ZeroDivisionError:
            warnings.warn(
                "Instability in Hawkes process: 1 - alpha/beta is zero or near zero.",
                UserWarning,
            )
            return 0.0

        # The original code provided this symmetric calculation:
        lambda_plus = lambda_avg
        lambda_minus = lambda_avg

        if lambda_plus == lambda_minus:
            warnings.warn(
                "Symmetric rates (lambda_plus == lambda_minus) result in a zero optimal spread by this formula.",
                UserWarning,
            )
            # The calculation is correct based on the given formula, but the result is zero.
            # return (lambda_plus - lambda_minus) / (lambda_plus + lambda_minus) * mid_price
            return 0.0

        # Keeping the structure of the original function's return statement,
        # even though with lambda_plus = lambda_minus, the result is 0.0.
        # return (lambda_plus - lambda_minus) / (lambda_plus + lambda_minus) * mid_price

        # A more realistic (though still simplified) example might use
        # the reciprocal of the rates, relating spread to liquidity risk:
        # return (1/lambda_minus - 1/lambda_plus) / (1/lambda_minus + 1/lambda_plus) * mid_price

        return 0.0
