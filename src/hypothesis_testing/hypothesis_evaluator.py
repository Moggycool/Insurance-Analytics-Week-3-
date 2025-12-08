class HypothesisEvaluator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def evaluate(self, p_value):
        return "Reject H₀" if p_value < self.alpha else "Fail to Reject H₀"
