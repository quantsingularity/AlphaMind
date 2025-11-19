# import QuantLib as ql


# class CVAcalculator:
#     def __init__(self, portfolio, default_probs):
#         self.portfolio = portfolio
#         self.default_probs = default_probs

#     def calculate_cva(self, discount_curve):
#         cva = 0.0
#         for trade in self.portfolio:
#             dates, exposures = trade["exposure_profile"]
#             for date, exposure in zip(dates, exposures):
#                 t = discount_curve.timeFromReference(date)
#                 surv_prob = np.exp(-self.default_probs["hazard_rate"] * t)
#                 cva += (
#                     (1 - self.default_probs["recovery_rate"])
#                     * exposure
#                     * discount_curve.discount(t)
#                     * self.default_probs["hazard_rate"]
#                     * surv_prob
                )
#         return cva
