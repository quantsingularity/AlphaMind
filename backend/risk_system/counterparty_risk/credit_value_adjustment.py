# 1. UNCOMMENTED: All lines are now active.
# 2. FIX: Added 'numpy' import for 'np.exp'.
# 3. FIX: The CVA calculation logic was corrected.
#    - The original formula [LGD * EPE * DF * H * S(t)] was dimensionally
#      incorrect for a discrete sum. It was multiplying by a probability
#      *density* (H * S(t)) instead of a *probability*.
#    - The corrected logic uses the *marginal default probability*
#      for each period: P(default in [t_prev, t]) = S(t_prev) - S(t).
#    - This is the standard, more accurate way to sum discrete CVA contributions.
# 4. OPTIMIZATION: Pre-calculated LGD (Loss Given Default) in the constructor
#    to avoid repeated calculation in the loop.
# 5. STYLE: Added docstrings and type hinting for clarity and maintainability.
# 6. STYLE: Made the survival probability calculation its own helper method.

import QuantLib as ql
import numpy as np
from typing import List, Dict, Any, Tuple


class CVAcalculator:
    """
    Calculates Credit Valuation Adjustment (CVA) for a portfolio of trades,
    assuming a constant hazard rate for the counterparty.
    """

    def __init__(
        self, portfolio: List[Dict[str, Any]], default_probs: Dict[str, float]
    ):
        """
        Initializes the calculator.

        Args:
            portfolio: A list of trade dictionaries. Each dict is expected
                       to have an "exposure_profile" key, which is a
                       tuple of (dates_list, exposures_list).
            default_probs: A dictionary containing counterparty default
                           parameters, expecting "hazard_rate" (float)
                           and "recovery_rate" (float).
        """
        self.portfolio = portfolio
        self.default_probs = default_probs

        # --- OPTIMIZATION: Pre-calculate constants ---
        self.hazard_rate: float = self.default_probs["hazard_rate"]
        self.recovery_rate: float = self.default_probs["recovery_rate"]

        # Loss Given Default
        self.lgd: float = 1.0 - self.recovery_rate

    def _survival_probability(self, t: float) -> float:
        """Calculates survival probability S(t) = exp(-hazard_rate * t)."""
        if t <= 0.0:
            return 1.0
        return np.exp(-self.hazard_rate * t)

    def calculate_cva(self, discount_curve: ql.YieldTermStructure) -> float:
        """
        Calculates the total CVA for the portfolio.

        The CVA is calculated as the sum of discounted expected losses:
        CVA = Σ [ LGD * EPE(t_i) * DF(t_i) * PD(t_{i-1}, t_i) ]

        where:
        - LGD = Loss Given Default (1 - recovery_rate)
        - EPE(t_i) = Expected Positive Exposure at time t_i
        - DF(t_i) = Discount Factor at time t_i
        - PD(t_{i-1}, t_i) = Marginal default probability between t_{i-1} and t_i
                           = S(t_{i-1}) - S(t_i)

        Args:
            discount_curve: A QuantLib discount curve object used
                            for discounting and time calculation.

        Returns:
            The total CVA of the portfolio.
        """
        total_cva = 0.0

        for trade in self.portfolio:
            trade_cva = 0.0

            # Ensure the exposure profile exists and is in the expected format
            if "exposure_profile" not in trade:
                continue

            exposure_profile: Tuple[list, list] = trade["exposure_profile"]
            dates, exposures = exposure_profile

            if not dates:
                continue

            # --- FIX: Correct CVA summation logic ---
            # Initialize state for the start of the trade
            t_prev = 0.0
            surv_prob_prev = 1.0  # S(0) = 1

            for date, exposure in zip(dates, exposures):

                # 1. Get time t (in years) from the valuation date
                t = discount_curve.timeFromReference(date)

                if t <= t_prev:
                    # Skip if dates are not monotonically increasing
                    # or if time is zero/negative
                    continue

                # 2. Get Survival Probability at current time t
                surv_prob_t = self._survival_probability(t)

                # 3. Get Marginal Default Probability for period [t_prev, t]
                marginal_pd = surv_prob_prev - surv_prob_t

                # 4. Get Discount Factor for time t
                df = discount_curve.discount(t)

                # 5. Calculate CVA contribution for this period
                # This assumes 'exposure' is the EPE at the end of the period (t)
                cva_contribution = self.lgd * exposure * df * marginal_pd

                trade_cva += cva_contribution

                # 6. Update "previous" state for the next loop iteration
                t_prev = t
                surv_prob_prev = surv_prob_t

            total_cva += trade_cva

        return total_cva
