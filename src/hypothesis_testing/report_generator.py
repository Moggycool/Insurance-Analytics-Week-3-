class ReportGenerator:
    def create(self, feature, metric, p_value, mean_a, mean_b):
        diff_pct = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0

        if p_value < 0.05:
            return (
                f"Reject H₀: {feature} significantly impacts {metric} (p={p_value:.4f}). "
                f"Group B differs by {diff_pct:.2f}%. "
                "Recommended: adjust segmentation or premiums."
            )
        return (
            f"Fail to Reject H₀: No statistically significant impact of {feature} "
            f"on {metric} (p={p_value:.4f})."
        )
