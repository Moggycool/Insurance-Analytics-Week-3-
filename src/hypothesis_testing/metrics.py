class MetricCalculator:
    def __init__(self, df):
        self.df = df

    def claim_frequency(self):
        return self.df["HasClaim"].mean()

    def claim_severity(self):
        claimed = self.df[self.df["HasClaim"] == 1]
        return claimed["ClaimSeverity"].mean() if not claimed.empty else 0

    def margin(self):
        return self.df["Margin"].mean()
