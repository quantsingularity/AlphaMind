# class LiquidityHawkesProcess:
#     def __init__(self, alpha=0.1, beta=0.3, mu=0.5):
#         self.alpha = alpha
#         self.beta = beta
#         self.mu = mu
#         self.intensity = mu

#     def simulate(self, T=1000):
#         events = []
#         t = 0
#         while t < T:
#             M = self.intensity
#             u = np.random.rand()
#             t += -np.log(u) / M
#             if t > T:
#                 break
#             events.append(t)
#             self.intensity = (
#                 self.mu
#                 + (self.intensity - self.mu) * np.exp(-self.beta * (t - events[-1]))
#                 + self.alpha
#             )
#         return np.array(events)

#     def forecast_optimal_spread(self, mid_price):
#         lambda_plus = self.mu / (1 - self.alpha / self.beta)
#         lambda_minus = self.mu / (1 - self.alpha / self.beta)
#         return (lambda_plus - lambda_minus) / (lambda_plus + lambda_minus) * mid_price
