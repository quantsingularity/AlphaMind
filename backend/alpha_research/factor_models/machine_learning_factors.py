# class AutoAlphaGenerator:
#     def __init__(self, n_factors=10, lookback=63):
#         self.autoencoder = StackedDenoisingAutoencoder(
#             layers=[256, 128, 64, n_factors], noise=0.1
        )
#         self.scaler = RobustScaler()
#         self.lookback = lookback

#     def generate_factors(self, returns):
#         X = self._create_rolling_dataset(returns)
#         X_scaled = self.scaler.fit_transform(X)
#         factors = self.autoencoder.encode(X_scaled)
#         return self._orthogonalize_factors(factors)

#     def _create_rolling_dataset(self, returns):
#         windows = [returns.shift(i) for i in range(self.lookback)]
#         return pd.concat(windows, axis=1).dropna()

#     def _orthogonalize_factors(self, factors):
#         pca = PCA(n_components=factors.shape[1])
#         return pca.fit_transform(factors)

#     def backtest_factors(self, factors, returns):
#         clf = LassoCV(cv=5)
#         clf.fit(factors, returns)
#         return {
            "r_squared": clf.score(factors, returns),
            "sharpe_ratio": self._factor_sharpe(clf.coef_),
            "turnover": self._calculate_turnover(clf.coef_),
        }
