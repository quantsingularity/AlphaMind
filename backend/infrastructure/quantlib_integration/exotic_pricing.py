class AutocallablePricer:
    def __init__(self, observation_dates, coupon_barriers):
        self.schedule = ql.Schedule(
            ql.Date(),
            observation_dates[-1],
            ql.Period("3M"),
            ql.Calendar(),
            ql.Unadjusted,
            ql.Unadjusted,
        )
        self.option = ql.AutocallableOption(
            coupon_barriers,
            observation_dates,
            ql.EuropeanExercise(observation_dates[-1]),
        )

    def calculate_knock_in_risk(self, paths):
        knock_in_count = 0
        for path in paths:
            if any(close < self.barrier for close in path):
                knock_in_count += 1
        return knock_in_count / len(paths)
