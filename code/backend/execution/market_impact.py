from typing import Any

import numpy as np
from scipy.integrate import solve_bvp


class OptimalExecution:

    def __init__(self, order_size: Any, liquidity_profile: Any) -> None:
        self.order_size = order_size
        self.liquidity = liquidity_profile

    def acati_strategy(self, risk_aversion: Any = 0.5) -> Any:
        """Almgren-Chriss optimal execution"""
        T = len(self.liquidity)
        eta = self.liquidity["elasticity"]
        gamma = self.liquidity["permanent_impact"]
        solution = solve_bvp(
            lambda t, x: self._ode_system(t, x, eta, gamma, risk_aversion),
            lambda xa, xb: self._bc(xa, xb, self.order_size),
            np.linspace(0, T, 100),
            np.zeros((2, 100)),
        )
        return solution.y[0]

    def _ode_system(self, t: Any, x: Any, eta: Any, gamma: Any, lambda_: Any) -> Any:
        """ODE system for optimal execution"""
        dqdt = x[1]
        dudt = 2 * eta / lambda_ * x[1] + gamma * dqdt
        return np.array([dqdt, dudt])

    def _bc(self, xa: Any, xb: Any, order_size: Any) -> Any:
        """Boundary conditions for optimal execution"""
        return np.array([xa[0] - order_size, xb[0]])
