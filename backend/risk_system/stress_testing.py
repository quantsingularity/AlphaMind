from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import multivariate_t, multivariate_normal


class ExtremeScenarioGenerator:

    def __init__(self, copula: Any = "t", tail_dependence: Any = 0.3) -> Any:
        self.copula = copula
        self.tail_dependence = tail_dependence

    def generate_crisis_scenarios(
        self, factor_correlations: Any, n_scenarios: Any = 1000
    ) -> Any:
        """
        Generate crisis scenarios using a t-copula with tail dependence.
        """
        crisis_corr = self._create_dependence_matrix(factor_correlations)
        if self.copula == "t":
            rv = multivariate_t(loc=np.zeros(len(crisis_corr)), shape=crisis_corr, df=3)
            return rv.rvs(n_scenarios)
        elif self.copula == "gaussian":
            rv = multivariate_normal(mean=np.zeros(len(crisis_corr)), cov=crisis_corr)
            return rv.rvs(n_scenarios)
        raise NotImplementedError(f"Copula type '{self.copula}' is not implemented.")

    def _create_dependence_matrix(self, base_corr: Any) -> Any:
        """
        Adjust normal correlations to crisis correlations using tail dependence.
        """
        base_corr = np.asarray(base_corr)
        base_corr.shape[0]
        if base_corr.shape[0] != base_corr.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        crisis_corr = np.copy(base_corr)
        np.fill_diagonal(crisis_corr, 1)
        crisis_corr = crisis_corr + (1 - crisis_corr) * self.tail_dependence
        return crisis_corr

    def apply_shock(self, portfolio_weights: Any, scenarios: Any) -> Any:
        """
        Apply scenarios to a portfolio.
        portfolio_weights: 1D vector of factor exposures
        scenarios: matrix of simulated factor shocks
        """
        portfolio_returns = scenarios @ portfolio_weights
        df = pd.DataFrame(
            {
                "VaR_99": [np.quantile(portfolio_returns, 0.01)],
                "Max_Drawdown": [portfolio_returns.min()],
                "Leverage_Impact": [np.linalg.norm(portfolio_returns, ord=2)],
                "Mean_Return": [portfolio_returns.mean()],
            }
        )
        return df
