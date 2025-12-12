from typing import Any
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az


class BayesianVaR:

    def __init__(self, returns: Any) -> Any:
        self.returns = np.asarray(returns)
        self.model = None
        self.trace = None

    def build_model(self) -> Any:
        T = len(self.returns)
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1, shape=3)
            P = pm.Dirichlet("P", a=np.ones((3, 3)), shape=(3, 3))
            pi_0 = pm.Dirichlet("pi_0", a=np.ones(3))
            states = pm.CategoricalMarkovChain("states", P=P, initial=pi_0, shape=T)
            omega = pm.HalfNormal("omega", sigma=0.1)
            alpha = pm.Beta("alpha", alpha=2, beta=5)
            beta = pm.Beta("beta", alpha=2, beta=5)
            eps = self.returns - mu[states]
            sigma2 = pt.zeros(T)
            sigma2 = pt.set_subtensor(sigma2[0], omega / (1 - alpha - beta))

            def garch_step(t, eps, sigma2, omega, alpha, beta):
                new_sigma2 = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                return new_sigma2

            sigma2, _ = pm.scan(
                fn=garch_step,
                sequences=[pt.arange(1, T)],
                outputs_info=[sigma2],
                non_sequences=[eps, omega, alpha, beta],
            )
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            pm.Normal("obs", mu=mu[states], sigma=sigma, observed=self.returns)
            trace = pm.sample(1500, tune=1000, target_accept=0.9, cores=4, chains=4)
        self.model = model
        self.trace = trace

    def calculate_var(self, alpha: Any = 0.05) -> Any:
        if self.model is None:
            raise ValueError("Call build_model() first.")
        with self.model:
            post = pm.sample_posterior_predictive(self.trace, var_names=["obs"])
        predictive = post["obs"].ravel()
        hdi = az.hdi(predictive, hdi_prob=1 - alpha)
        var_level = hdi[0]
        return var_level
