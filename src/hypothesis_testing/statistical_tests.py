import numpy as np
import scipy.stats as stats


class StatisticalTester:
    def chi_square(self, group_a, group_b, col):
        """
        Performs Chi-square test.
        Falls back to Fisher's Exact Test if table contains zeros.
        """
        a_count = group_a[col].sum()
        b_count = group_b[col].sum()

        a_total = len(group_a)
        b_total = len(group_b)

        # Build contingency table
        table = np.array([
            [a_count, a_total - a_count],
            [b_count, b_total - b_count]
        ])

        # If any expected frequencies are zero, use Fisher's Exact Test
        if (table == 0).any():
            odds_ratio, p_value = stats.fisher_exact(table)
            return odds_ratio, p_value

        chi2, p_value, _, _ = stats.chi2_contingency(table)
        return chi2, p_value

    def t_test(self, group_a, group_b, col):
        """
        Performs independent two-sample t-test.
        """
        stat, p_value = stats.ttest_ind(
            group_a[col],
            group_b[col],
            nan_policy="omit",
            equal_var=False
        )
        return stat, p_value
