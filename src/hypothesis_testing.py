"""
Hypothesis Testing Module for Insurance Analytics
Task 3: Statistical validation of risk drivers for segmentation strategy
Consolidated version with all functionality in one module (< 1000 lines)
"""

# Standard library imports
import warnings
import os
import json
import datetime
from typing import Optional, Dict, Any

# Third party imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower

warnings.filterwarnings('ignore')


class HypothesisTester:
    """
    Comprehensive hypothesis testing class for insurance risk analysis.
    Tests key hypotheses about risk drivers for segmentation strategy.
    """

    def __init__(self, df):
        """
        Initialize with insurance data DataFrame.

        Args:
            df: Pandas DataFrame with processed insurance data
        """
        self.df = df.copy()
        self.results = {}
        self.alpha = 0.05  # Significance level

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def calculate_risk_metrics(self):
        """Calculate Claim Frequency and Claim Severity metrics."""

        # Claim Frequency: Proportion of policies with at least one claim
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)

        # Claim Severity: Average claim amount when claim occurs
        claim_mask = self.df['TotalClaims'] > 0
        self.df['ClaimSeverity'] = 0
        self.df.loc[claim_mask,
                    'ClaimSeverity'] = self.df.loc[claim_mask, 'TotalClaims']

        # Margin: Profit = TotalPremium - TotalClaims
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

        # Profit Margin Percentage
        self.df['ProfitMarginPct'] = self.df['Margin'] / \
            self.df['TotalPremium'].replace(0, np.nan)

        print("✅ Risk metrics calculated:")
        print(f"   • Claim Frequency: {self.df['HasClaim'].mean():.2%}")
        print(
            f"   • Average Claim Severity: ${self.df.loc[claim_mask, 'ClaimSeverity'].mean():,.2f}")
        print(f"   • Total Margin: ${self.df['Margin'].sum():,.2f}")
        print(
            f"   • Average Profit Margin: {self.df['ProfitMarginPct'].mean():.2%}")

        return self.df

    def calculate_effect_size(self, group1_data, group2_data):
        """
        Calculate Cohen's d effect size.
        """
        n1, n2 = len(group1_data), len(group2_data)
        s1, s2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std

        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            interpretation = "Negligible"
        elif d_abs < 0.5:
            interpretation = "Small"
        elif d_abs < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"

        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'mean_diff': np.mean(group1_data) - np.mean(group2_data),
            'pooled_std': pooled_std
        }

    def power_analysis(self, group1_data, group2_data, effect_size=None):
        """
        Perform power analysis for t-test.
        """
        n1, n2 = len(group1_data), len(group2_data)

        if effect_size is None:
            effect_size_dict = self.calculate_effect_size(
                group1_data, group2_data)
            effect_size_val = abs(effect_size_dict['cohens_d'])
        else:
            effect_size_val = abs(effect_size)

        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(
            effect_size=effect_size_val,
            nobs1=n1,
            alpha=self.alpha,
            ratio=n2/n1
        )

        return {
            'power': power,
            'effect_size': effect_size_val,
            'n1': n1,
            'n2': n2,
            'alpha': self.alpha
        }

    # ============================================================================
    # HYPOTHESIS TESTING METHODS
    # ============================================================================

    def test_province_risk_differences(self):
        """
        Test H₀: There are no risk differences across provinces.
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 1: PROVINCE RISK DIFFERENCES")
        print("="*60)

        if 'Province' not in self.df.columns:
            print("⚠️ 'Province' column not found in data")
            return None

        if 'HasClaim' not in self.df.columns:
            self.calculate_risk_metrics()

        province_data = self.df[['Province',
                                 'HasClaim', 'ClaimSeverity']].dropna()
        province_counts = province_data['Province'].value_counts()
        valid_provinces = province_counts[province_counts >= 30].index
        province_data = province_data[province_data['Province'].isin(
            valid_provinces)]

        if len(valid_provinces) < 2:
            print("⚠️ Insufficient provinces for comparison")
            return None

        print(
            f"📊 Analyzing {len(province_data):,} policies across {len(valid_provinces)} provinces")
        results = {'test_1a': {}, 'test_1b': {}}

        # TEST 1A: CLAIM FREQUENCY BY PROVINCE
        print("\n📊 Test 1A: Claim Frequency by Province (Chi-square test)")
        print("-" * 40)

        contingency = pd.crosstab(
            province_data['Province'], province_data['HasClaim'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        _ = expected

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        if cramers_v < 0.1:
            effect_interpretation = "Negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "Small"
        elif cramers_v < 0.5:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"

        results['test_1a'] = {
            'test_name': 'Province Claim Frequency Differences',
            'null_hypothesis': 'There are no claim frequency differences across provinces',
            'test_type': 'Chi-square Test',
            'statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'effect_size': cramers_v,
            'effect_interpretation': effect_interpretation,
            'n_provinces': len(valid_provinces),
            'n_observations': n,
            'reject_null': p_value < self.alpha
        }

        # TEST 1B: CLAIM SEVERITY BY PROVINCE
        print("\n📊 Test 1B: Claim Severity by Province (ANOVA)")
        print("-" * 40)

        claims_data = province_data[province_data['HasClaim'] == 1]

        if len(claims_data) < 30:
            print("⚠️ Insufficient claim data for severity analysis")
            results['test_1b'] = None
        else:
            groups = []
            group_names = []
            for province in valid_provinces[:10]:
                province_claims = claims_data[claims_data['Province']
                                              == province]['ClaimSeverity']
                if len(province_claims) >= 5:
                    groups.append(province_claims)
                    group_names.append(province)

            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)

                ss_between = 0
                ss_total = 0
                grand_mean = np.concatenate(groups).mean()

                for group in groups:
                    ss_between += len(group) * (group.mean() - grand_mean) ** 2
                    ss_total += ((group - grand_mean) ** 2).sum()

                eta_squared = ss_between / ss_total

                if eta_squared < 0.01:
                    effect_interpretation = "Negligible"
                elif eta_squared < 0.06:
                    effect_interpretation = "Small"
                elif eta_squared < 0.14:
                    effect_interpretation = "Medium"
                else:
                    effect_interpretation = "Large"

                results['test_1b'] = {
                    'test_name': 'Province Claim Severity Differences',
                    'null_hypothesis': 'There are no claim severity differences across provinces',
                    'test_type': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'effect_size': eta_squared,
                    'effect_interpretation': effect_interpretation,
                    'n_groups': len(groups),
                    'n_observations': sum(len(g) for g in groups),
                    'reject_null': p_value < self.alpha
                }

                if p_value < self.alpha and len(groups) > 2:
                    print("   Performing Tukey HSD post-hoc test...")
                    tukey_data = pd.DataFrame({
                        'ClaimSeverity': np.concatenate(groups),
                        'Province': np.concatenate([[name]*len(g) for name, g in zip(group_names, groups)])
                    })
                    tukey_result = pairwise_tukeyhsd(tukey_data['ClaimSeverity'],
                                                     tukey_data['Province'],
                                                     alpha=self.alpha)
                    significant_pairs = []
                    for i in range(len(tukey_result.groupsunique)):
                        for j in range(i+1, len(tukey_result.groupsunique)):
                            if tukey_result.reject[i, j]:
                                pair = (
                                    tukey_result.groupsunique[i], tukey_result.groupsunique[j])
                                significant_pairs.append(pair)
                    if significant_pairs:
                        results['test_1b']['post_hoc'] = significant_pairs[:5]
                    else:
                        results['test_1b']['post_hoc'] = []
            else:
                results['test_1b'] = None

        self.results['province_tests'] = results
        return results

    def test_zipcode_risk_differences(self):
        """
        Test H₀: There are no risk differences between zip codes.
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 2: ZIPCODE RISK DIFFERENCES")
        print("="*60)

        zipcode_cols = [col for col in self.df.columns
                        if any(keyword in col.lower() for keyword in ['postal', 'zip', 'code'])]

        if not zipcode_cols:
            print("⚠️ No zipcode column found in data")
            return None

        zipcode_col = zipcode_cols[0]
        print(f"📊 Using column '{zipcode_col}' for zipcode analysis")

        if 'HasClaim' not in self.df.columns:
            self.calculate_risk_metrics()

        zipcode_data = self.df[[zipcode_col, 'HasClaim']].dropna()
        zipcode_data['ZipPrefix'] = zipcode_data[zipcode_col].astype(
            str).str.extract(r'(\d{3,4})')[0]
        zipcode_data = zipcode_data[zipcode_data['ZipPrefix'].notna()]

        zip_counts = zipcode_data['ZipPrefix'].value_counts()
        valid_zips = zip_counts[zip_counts >= 50].index
        zipcode_data = zipcode_data[zipcode_data['ZipPrefix'].isin(valid_zips)]

        if len(valid_zips) < 2:
            print("⚠️ Insufficient zipcodes for comparison")
            return None

        print(
            f"📊 Analyzing {len(zipcode_data):,} policies across {len(valid_zips)} zipcode areas")
        top_zips = zip_counts.nlargest(20).index
        zipcode_data = zipcode_data[zipcode_data['ZipPrefix'].isin(top_zips)]

        print("\n📊 Test: Claim Frequency by Zipcode (Chi-square test)")
        print("-" * 40)

        contingency = pd.crosstab(
            zipcode_data['ZipPrefix'], zipcode_data['HasClaim'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        _ = expected

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        if cramers_v < 0.1:
            effect_interpretation = "Negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "Small"
        elif cramers_v < 0.5:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"

        claim_freq_by_zip = zipcode_data.groupby(
            'ZipPrefix')['HasClaim'].mean().sort_values(ascending=False)

        results = {
            'test_name': 'Zipcode Claim Frequency Differences',
            'null_hypothesis': 'There are no claim frequency differences between zip codes',
            'test_type': 'Chi-square Test',
            'statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'effect_size': cramers_v,
            'effect_interpretation': effect_interpretation,
            'n_zipcodes': len(top_zips),
            'n_observations': n,
            'reject_null': p_value < self.alpha,
            'top_5_high_risk': claim_freq_by_zip.head(5).to_dict(),
            'top_5_low_risk': claim_freq_by_zip.tail(5).to_dict()
        }

        self.results['zipcode_risk_test'] = results
        return results

    def test_zipcode_profit_differences(self):
        """
        Test H₀: There is no significant margin difference between zip codes.
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 3: ZIPCODE PROFIT DIFFERENCES")
        print("="*60)

        zipcode_cols = [col for col in self.df.columns
                        if any(keyword in col.lower() for keyword in ['postal', 'zip', 'code'])]

        if not zipcode_cols:
            print("⚠️ No zipcode column found in data")
            return None

        zipcode_col = zipcode_cols[0]

        if 'Margin' not in self.df.columns:
            self.calculate_risk_metrics()

        profit_data = self.df[[zipcode_col,
                               'Margin', 'ProfitMarginPct']].dropna()
        profit_data['ZipPrefix'] = profit_data[zipcode_col].astype(
            str).str.extract(r'(\d{3,4})')[0]
        profit_data = profit_data[profit_data['ZipPrefix'].notna()]

        zip_counts = profit_data['ZipPrefix'].value_counts()
        valid_zips = zip_counts[zip_counts >= 30].index
        profit_data = profit_data[profit_data['ZipPrefix'].isin(valid_zips)]

        if len(valid_zips) < 2:
            print("⚠️ Insufficient zipcodes for comparison")
            return None

        print(
            f"📊 Analyzing {len(profit_data):,} policies across {len(valid_zips)} zipcode areas")
        top_zips = zip_counts.nlargest(15).index
        profit_data = profit_data[profit_data['ZipPrefix'].isin(top_zips)]

        print("\n📊 Test: Margin by Zipcode (ANOVA)")
        print("-" * 40)

        groups = []
        for zipcode in top_zips:
            zip_margin = profit_data[profit_data['ZipPrefix']
                                     == zipcode]['Margin']
            if len(zip_margin) >= 5:
                groups.append(zip_margin)

        if len(groups) >= 2:
            f_stat, p_value = f_oneway(*groups)

            ss_between = 0
            ss_total = 0
            grand_mean = np.concatenate(groups).mean()

            for group in groups:
                ss_between += len(group) * (group.mean() - grand_mean) ** 2
                ss_total += ((group - grand_mean) ** 2).sum()

            eta_squared = ss_between / ss_total

            if eta_squared < 0.01:
                effect_interpretation = "Negligible"
            elif eta_squared < 0.06:
                effect_interpretation = "Small"
            elif eta_squared < 0.14:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"

            margin_by_zip = profit_data.groupby(
                'ZipPrefix')['Margin'].mean().sort_values(ascending=False)
            profit_margin_by_zip = profit_data.groupby(
                'ZipPrefix')['ProfitMarginPct'].mean().sort_values(ascending=False)

            results = {
                'test_name': 'Zipcode Profit Differences',
                'null_hypothesis': 'There is no significant margin difference between zip codes',
                'test_type': 'One-way ANOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'effect_size': eta_squared,
                'effect_interpretation': effect_interpretation,
                'n_groups': len(groups),
                'n_observations': sum(len(g) for g in groups),
                'reject_null': p_value < self.alpha,
                'top_5_high_margin': margin_by_zip.head(5).to_dict(),
                'top_5_low_margin': margin_by_zip.tail(5).to_dict(),
                'top_5_high_profit_margin': profit_margin_by_zip.head(5).to_dict(),
                'top_5_low_profit_margin': profit_margin_by_zip.tail(5).to_dict()
            }

            self.results['zipcode_profit_test'] = results
            return results

        return None

    def test_gender_risk_differences(self):
        """
        Test H₀: There is no significant risk difference between Women and Men.
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 4: GENDER RISK DIFFERENCES")
        print("="*60)

        if 'Gender' not in self.df.columns:
            print("⚠️ 'Gender' column not found in data")
            return None

        if 'HasClaim' not in self.df.columns:
            self.calculate_risk_metrics()

        gender_data = self.df[['Gender', 'HasClaim', 'ClaimSeverity']].dropna()
        gender_mapping = {
            'M': 'Male', 'MALE': 'Male', 'male': 'Male',
            'F': 'Female', 'FEMALE': 'Female', 'female': 'Female',
            '1': 'Male', '0': 'Female'
        }

        gender_data['Gender'] = gender_data['Gender'].astype(
            str).str.upper().map(gender_mapping)
        gender_data = gender_data[gender_data['Gender'].isin(
            ['Male', 'Female'])]

        if len(gender_data) < 100:
            print("⚠️ Insufficient gender data for analysis")
            return None

        print(f"📊 Analyzing {len(gender_data):,} policies by gender")
        print(
            f"   • Male: {len(gender_data[gender_data['Gender'] == 'Male']):,}")
        print(
            f"   • Female: {len(gender_data[gender_data['Gender'] == 'Female']):,}")

        results = {'test_4a': {}, 'test_4b': {}}

        # TEST 4A: CLAIM FREQUENCY BY GENDER
        print("\n📊 Test 4A: Claim Frequency by Gender (Chi-square test)")
        print("-" * 40)

        contingency = pd.crosstab(
            gender_data['Gender'], gender_data['HasClaim'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        _ = expected

        male_claim_rate = contingency.loc['Male',
                                          1] / contingency.loc['Male'].sum()
        female_claim_rate = contingency.loc['Female',
                                            1] / contingency.loc['Female'].sum()
        risk_ratio = female_claim_rate / male_claim_rate if male_claim_rate > 0 else np.nan

        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        results['test_4a'] = {
            'test_name': 'Gender Claim Frequency Differences',
            'null_hypothesis': 'There is no claim frequency difference between Women and Men',
            'test_type': 'Chi-square Test',
            'statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'male_claim_rate': male_claim_rate,
            'female_claim_rate': female_claim_rate,
            'risk_ratio': risk_ratio,
            'effect_size': cramers_v,
            'n_observations': n,
            'reject_null': p_value < self.alpha
        }

        # TEST 4B: CLAIM SEVERITY BY GENDER
        print("\n📊 Test 4B: Claim Severity by Gender")
        print("-" * 40)

        claims_data = gender_data[gender_data['HasClaim'] == 1]

        if len(claims_data) < 30:
            print("⚠️ Insufficient claim data for severity analysis")
            results['test_4b'] = None
        else:
            male_severity = claims_data[claims_data['Gender']
                                        == 'Male']['ClaimSeverity']
            female_severity = claims_data[claims_data['Gender']
                                          == 'Female']['ClaimSeverity']

            print(f"   • Male claims: {len(male_severity):,}")
            print(f"   • Female claims: {len(female_severity):,}")

            if len(male_severity) >= 10 and len(female_severity) >= 10:
                _, p_male_norm = stats.shapiro(male_severity)
                _, p_female_norm = stats.shapiro(female_severity)
                _, p_levene = stats.levene(male_severity, female_severity)

                assumptions = {
                    'male_normality': p_male_norm > 0.05,
                    'female_normality': p_female_norm > 0.05,
                    'equal_variance': p_levene > 0.05
                }

                if all(assumptions.values()):
                    test_type = "Independent t-test"
                    stat, p_value = ttest_ind(
                        male_severity, female_severity, equal_var=True)
                elif assumptions['equal_variance']:
                    test_type = "Welch's t-test"
                    stat, p_value = ttest_ind(
                        male_severity, female_severity, equal_var=False)
                else:
                    test_type = "Mann-Whitney U test"
                    stat, p_value = mannwhitneyu(
                        male_severity, female_severity, alternative='two-sided')

                effect_size = self.calculate_effect_size(
                    male_severity, female_severity)
                power_results = self.power_analysis(
                    male_severity, female_severity, effect_size['cohens_d'])

                results['test_4b'] = {
                    'test_name': 'Gender Claim Severity Differences',
                    'null_hypothesis': 'There is no claim severity difference between Women and Men',
                    'test_type': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'assumptions': assumptions,
                    'effect_size': effect_size,
                    'power_analysis': power_results,
                    'male_mean': male_severity.mean(),
                    'female_mean': female_severity.mean(),
                    'male_std': male_severity.std(),
                    'female_std': female_severity.std(),
                    'n_male': len(male_severity),
                    'n_female': len(female_severity),
                    'reject_null': p_value < self.alpha
                }
            else:
                results['test_4b'] = None

        self.results['gender_tests'] = results
        return results

    # ============================================================================
    # EXECUTION AND REPORTING METHODS
    # ============================================================================

    def run_all_tests(self):
        """Run all hypothesis tests sequentially."""
        print("="*70)
        print("RUNNING ALL HYPOTHESIS TESTS")
        print("="*70)

        self.calculate_risk_metrics()

        tests = [
            ('Province Risk Differences', self.test_province_risk_differences),
            ('Zipcode Risk Differences', self.test_zipcode_risk_differences),
            ('Zipcode Profit Differences', self.test_zipcode_profit_differences),
            ('Gender Risk Differences', self.test_gender_risk_differences)
        ]

        for test_name, test_func in tests:
            print(f"\n🚀 Running: {test_name}")
            print("-" * 40)
            try:
                result = test_func()
                if result:
                    print(f"✅ {test_name} completed")
                else:
                    print(f"⚠️ {test_name} could not be completed")
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"❌ Error in {test_name}: {e}")

        return self.results

    def generate_summary_report(self):
        """Generate a comprehensive summary report of all tests."""
        print("\n" + "="*70)
        print("HYPOTHESIS TESTING SUMMARY REPORT")
        print("="*70)

        summary_data = []

        # Process province tests
        if 'province_tests' in self.results:
            province_results = self.results['province_tests']
            for test_key in ['test_1a', 'test_1b']:
                if test_key in province_results and province_results[test_key]:
                    test = province_results[test_key]
                    summary_data.append({
                        'Hypothesis': test['test_name'],
                        'Test': test['test_type'],
                        'P-value': f"{test['p_value']:.6f}",
                        'Significant': 'Yes' if test['reject_null'] else 'No',
                        'Conclusion': 'Reject H₀' if test['reject_null'] else 'Fail to reject H₀',
                        'Effect Size': f"{test.get('effect_size', 0):.3f}",
                        'Business Implication': ('Regional segmentation recommended'
                                                 if test['reject_null']
                                                 else 'Regional segmentation not supported')
                    })

        # Process other tests
        for key in ['zipcode_risk_test', 'zipcode_profit_test']:
            if key in self.results and self.results[key]:
                test = self.results[key]
                summary_data.append({
                    'Hypothesis': test['test_name'],
                    'Test': test['test_type'],
                    'P-value': f"{test['p_value']:.6f}",
                    'Significant': 'Yes' if test['reject_null'] else 'No',
                    'Conclusion': 'Reject H₀' if test['reject_null'] else 'Fail to reject H₀',
                    'Effect Size': f"{test.get('effect_size', 0):.3f}",
                    'Business Implication': ('Geographic segmentation recommended'
                                             if test['reject_null']
                                             else 'Geographic segmentation not supported')
                })

        # Process gender tests
        if 'gender_tests' in self.results:
            gender_results = self.results['gender_tests']
            for test_key in ['test_4a', 'test_4b']:
                if test_key in gender_results and gender_results[test_key]:
                    test = gender_results[test_key]
                    effect_size = (test.get('effect_size', {}).get('cohens_d', 0)
                                   if isinstance(test.get('effect_size'), dict)
                                   else test.get('effect_size', 0))
                    summary_data.append({
                        'Hypothesis': test['test_name'],
                        'Test': test['test_type'],
                        'P-value': f"{test['p_value']:.6f}",
                        'Significant': 'Yes' if test['reject_null'] else 'No',
                        'Conclusion': 'Reject H₀' if test['reject_null'] else 'Fail to reject H₀',
                        'Effect Size': f"{effect_size:.3f}",
                        'Business Implication': ('Gender-based segmentation considered'
                                                 if test['reject_null']
                                                 else 'Gender-neutral approach supported')
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n📋 SUMMARY OF HYPOTHESIS TESTS:")
            print("="*80)
            print(summary_df.to_string(index=False))

            significant_count = sum(
                1 for item in summary_data if item['Significant'] == 'Yes')
            total_count = len(summary_data)
            print(
                f"\n📊 OVERALL: {significant_count}/{total_count} hypotheses rejected (significant)")

            print("\n🎯 BUSINESS RECOMMENDATIONS:")
            print("-" * 40)

            for item in summary_data:
                if item['Significant'] == 'Yes':
                    if 'Province' in item['Hypothesis']:
                        print("• Implement province-based risk segmentation")
                    elif 'Zipcode' in item['Hypothesis'] and 'Profit' in item['Hypothesis']:
                        print("• Target marketing to high-profit zipcode areas")
                    elif 'Zipcode' in item['Hypothesis']:
                        print("• Consider zipcode-level risk assessment")
                    elif 'Gender' in item['Hypothesis']:
                        print("• Review gender impact within regulatory guidelines")

            if significant_count == 0:
                print(
                    "• No significant differences found - consider alternative segmentation variables")
                print("• Focus on other customer attributes for segmentation")

            return summary_df

        print("⚠️ No test results available for summary")
        return None

    def save_results(self, output_dir='reports/hypothesis_tests'):
        """Save all test results to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results as JSON
        results_file = os.path.join(
            output_dir, f'hypothesis_results_{timestamp}.json')

        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        serializable_results = convert_for_json(self.results)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"✅ Detailed results saved to: {results_file}")

        # Generate and save summary report
        summary_df = self.generate_summary_report()
        if summary_df is not None:
            summary_file = os.path.join(
                output_dir, f'hypothesis_summary_{timestamp}.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary report saved to: {summary_file}")

        # Generate markdown report
        report_file = os.path.join(
            output_dir, f'hypothesis_report_{timestamp}.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Hypothesis Testing Report\n\n")
            f.write(
                f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Significance Level (α)**: {self.alpha}\n\n")

            # Count significant results
            significant_count = 0
            total_count = 0
            for tests in self.results.values():
                if isinstance(tests, dict):
                    for test_result in tests.values():
                        if test_result and isinstance(test_result, dict):
                            total_count += 1
                            if test_result.get('reject_null', False):
                                significant_count += 1

            f.write(f"**Total Tests**: {total_count}\n")
            f.write(f"**Significant Findings**: {significant_count}\n")
            f.write(
                f"**Non-significant Findings**: {total_count - significant_count}\n\n")

            f.write("## Recommendations\n\n")
            if significant_count > 0:
                f.write("Based on significant findings:\n\n")
                f.write(
                    "1. **Implement segmentation** for variables showing significant differences\n")
                f.write(
                    "2. **Develop targeted marketing** strategies for different segments\n")
                f.write("3. **Adjust pricing models** to reflect risk differences\n")
                f.write("4. **Monitor segment performance** regularly\n")
            else:
                f.write(
                    "No significant differences were found across tested dimensions.\n\n")
                f.write("Consider:\n\n")
                f.write("1. Exploring alternative segmentation variables\n")
                f.write("2. More granular geographic analysis\n")
                f.write("3. Behavioral or temporal factors\n")
                f.write("4. Combined variable segmentation\n")

        print(f"✅ Markdown report saved to: {report_file}")
        return results_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Hypothesis Testing Module for Insurance Analytics")
    print("This module should be imported and used with your data.")
