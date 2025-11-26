import QuantLib as ql


class StochasticVolatilitySurface:
    def __init__(self, calibration_data):
        self.calibration_data = calibration_data
        self.helpers = []
        self.model = ql.HestonModel(
            ql.HestonProcess(
                ql.YieldTermStructureHandle(
                    ql.FlatForward(0, ql.TARGET(), 0.01, ql.Actual365Fixed())
                ),
                calibration_data["spot"],
                calibration_data["v0"],
                calibration_data["kappa"],
                calibration_data["theta"],
                calibration_data["sigma"],
                calibration_data["rho"],
            )
        )

    def calibrate(self):
        optimization_method = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        self.model.calibrate(
            self.helpers,
            optimization_method,
            ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8),
        )

    def calculate_arbitrage_free_surface(self):
        heston_process = self.model.process()
        return ql.HestonModelHelper(
            heston_process,
            ql.BicubicFixedLocalVolSurface(
                ql.BlackVarianceSurface(
                    ql.Date(),
                    ql.Calendar(),
                    self.calibration_data["dates"],
                    self.calibration_data["strikes"],
                    self.calibration_data["vol_matrix"],
                    ql.Actual365Fixed(),
                )
            ),
        )
