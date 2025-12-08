from src.hypothesis_testing.runner import HypothesisRunner

if __name__ == "__main__":
    file_path = "data/processed/processed_MachineLearningRating_v3.csv"

    runner = HypothesisRunner(file_path)

    tests = [
        ("Province", None, None, "HasClaim"),
        ("PostalCode", None, None, "ClaimSeverity"),
        ("PostalCode", None, None, "Margin"),
        ("Gender", None, None, "HasClaim"),
    ]

    print("\n--- Hypothesis Test Results ---\n")

    for feature, g1, g2, metric in tests:
        print(runner.run(feature, g1, g2, metric))
