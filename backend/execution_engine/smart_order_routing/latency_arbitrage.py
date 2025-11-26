class LatencyOptimizedRouter:
    def __init__(self, venue_latencies):
        self.latencies = venue_latencies

    def route_order(self, order, market_data):
        adjusted_prices = {}
        for venue in self.latencies:
            future_price = self._predict_price(
                market_data[venue], self.latencies[venue]
            )
            adjusted_prices[venue] = future_price

        best_venue = max(adjusted_prices, key=lambda x: x["adjusted_price"])
        return {
            "venue": best_venue,
            "price": adjusted_prices[best_venue],
            "size": order["size"],
            "valid_for": self.latencies[best_venue] * 0.8,
        }
