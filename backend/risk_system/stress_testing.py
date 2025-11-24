# import numpy as np
# import pandas as pd
# from scipy.stats import multivariate_t


# class ExtremeScenarioGenerator:
#     def __init__(self, copula="t", tail_dependence=0.3):
#         self.copula = copula
#         self.tail_dependence = tail_dependence

#     def generate_crisis_scenarios(self, factor_correlations):
#         if self.copula == "t":
#             rv = multivariate_t(
#                 df=3, shape=self._create_dependence_matrix(factor_correlations)
#         return rv.rvs(1000)

#     def _create_dependence_matrix(self, base_corr):
#         n = len(base_corr)
#         crisis_corr = np.copy(base_corr)
#         np.fill_diagonal(crisis_corr, 1)
#         crisis_corr = crisis_corr + (1 - crisis_corr) * self.tail_dependence
#         return crisis_corr

#     def apply_shock(self, portfolio, scenario):
#         shocked_returns = portfolio @ scenario.T
#         return pd.DataFrame(
            {
                "VaR_99": shocked_returns.quantile(0.01),
                "Max_Drawdown": shocked_returns.min(),
                "Leverage_Impact": np.linalg.norm(shocked_returns, ord=2),
            }
