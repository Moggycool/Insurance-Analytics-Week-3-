class SegmentationEngine:
    def __init__(self, df):
        self.df = df

    def top_categories(self, feature, n=2):
        """
        Get top N most frequent categories
        """
        return (
            self.df[feature]
            .value_counts()
            .nlargest(n)
            .index
            .tolist()
        )

    def split_auto(self, feature):
        """
        Automatically pick the two largest groups for comparison
        """
        cats = self.top_categories(feature, 2)

        if len(cats) < 2:
            return None, None, None, None

        group_a, group_b = cats

        A = self.df[self.df[feature] == group_a]
        B = self.df[self.df[feature] == group_b]

        return A, B, group_a, group_b
