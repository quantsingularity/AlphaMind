class OptimalExecution:

    def __init__(self, order_size: Any, liquidity_profile: Any) -> Any:
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
            np.zeros(100),
        )
        return solution.y[0]

    def _ode_system(self, t: Any, x: Any, eta: Any, gamma: Any, lambda_: Any) -> Any:
        dqdt = x[1]
        dudt = 2 * eta / lambda_ * x[1] + gamma * dqdt
        return [dqdt, dudt]
