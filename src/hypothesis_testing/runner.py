from .data_loader import DataLoader
from .segmentation import SegmentationEngine
from .statistical_tests import StatisticalTester
from typing import Optional, Dict, Any


class HypothesisRunner:
    def __init__(self, file_path: Optional[str] = None, *args, **kwargs):
        """
        VS-Code / type-checker friendly constructor
        """
        self.file_path = (
            file_path
            or (args[0] if len(args) > 0 else None)
            or kwargs.get("file_path")
            or "data/processed/processed_MachineLearningRating_v3.csv"
        )

    def run(self, *args) -> Dict[str, Any]:
        """
        Safe runner that handles any argument shape
        """

        # Support both old and new call styles
        if len(args) == 4:
            feature = args[0]
            metric_col = args[3]
        elif len(args) == 2:
            feature = args[0]
            metric_col = args[1]
        else:
            return {
                "decision": "Invalid",
                "summary": "Invalid test configuration."
            }

        loader = DataLoader(self.file_path)
        df = loader.load()

        df = loader.create_metrics()

        seg = SegmentationEngine(df)
        tester = StatisticalTester()

        A, B, group_a, group_b = seg.split_auto(feature)

        if A is None or B is None or len(A) < 30 or len(B) < 30:
            return {
                "feature": feature,
                "metric": metric_col,
                "decision": "Insufficient Data",
                "summary": f"Not enough data to compare groups for {feature}."
            }

        if metric_col == "HasClaim":
            stat, p = tester.chi_square(A, B, metric_col)
        else:
            stat, p = tester.t_test(A, B, metric_col)

        if p < 0.05:
            return {
                "feature": feature,
                "metric": metric_col,
                "group_a": group_a,
                "group_b": group_b,
                "p_value": float(p),
                "decision": "Reject H₀",
                "summary": (
                    f"Significant impact of {feature} on {metric_col} "
                    f"between {group_a} and {group_b} (p={p:.4f})"
                )
            }
        else:
            return {
                "feature": feature,
                "metric": metric_col,
                "group_a": group_a,
                "group_b": group_b,
                "p_value": float(p),
                "decision": "Fail to Reject H₀",
                "summary": (
                    f"No statistically significant impact of {feature} on {metric_col} "
                    f"between {group_a} and {group_b} (p={p:.4f})"
                )
            }
