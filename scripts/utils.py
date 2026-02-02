"""
Utility functions for statistics practice
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_data(data):
    """
    Print comprehensive summary statistics for a dataset
    
    Parameters:
    -----------
    data : array-like or pd.Series
        The data to summarize
    """
    data = np.array(data)
    
    print("=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(f"Count:          {len(data)}")
    print(f"Mean:           {np.mean(data):.4f}")
    print(f"Median:         {np.median(data):.4f}")
    print(f"Mode:           {stats.mode(data, keepdims=True)[0][0]:.4f}")
    print(f"Std Dev:        {np.std(data, ddof=1):.4f}")
    print(f"Variance:       {np.var(data, ddof=1):.4f}")
    print(f"Min:            {np.min(data):.4f}")
    print(f"25th %ile:      {np.percentile(data, 25):.4f}")
    print(f"50th %ile:      {np.percentile(data, 50):.4f}")
    print(f"75th %ile:      {np.percentile(data, 75):.4f}")
    print(f"Max:            {np.max(data):.4f}")
    print(f"Range:          {np.ptp(data):.4f}")
    print(f"IQR:            {np.percentile(data, 75) - np.percentile(data, 25):.4f}")
    print(f"Skewness:       {stats.skew(data):.4f}")
    print(f"Kurtosis:       {stats.kurtosis(data):.4f}")
    print("=" * 50)


def plot_distribution(data, title="Distribution Plot", bins=30):
    """
    Create a comprehensive distribution plot with histogram and box plot
    
    Parameters:
    -----------
    data : array-like
        The data to plot
    title : str
        Title for the plot
    bins : int
        Number of bins for histogram
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(data, bins=bins, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
    axes[0, 0].axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.2f}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(data, vert=True)
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Box Plot')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Density plot
    axes[1, 1].hist(data, bins=bins, density=True, alpha=0.5, edgecolor='black')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Density Plot')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    axes[1, 1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    axes[1, 1].legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def check_normality(data, alpha=0.05):
    """
    Perform normality tests on data
    
    Parameters:
    -----------
    data : array-like
        The data to test
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Dictionary with test results
    """
    data = np.array(data)
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(data)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    
    print("=" * 50)
    print("NORMALITY TESTS")
    print("=" * 50)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value:   {shapiro_p:.4f}")
    print(f"  Normal?    {'Yes' if shapiro_p > alpha else 'No'} (α={alpha})")
    print()
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value:   {ks_p:.4f}")
    print(f"  Normal?    {'Yes' if ks_p > alpha else 'No'} (α={alpha})")
    print()
    print(f"Anderson-Darling Test:")
    print(f"  Statistic: {anderson_result.statistic:.4f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    print("=" * 50)
    
    return {
        'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
        'ks': {'statistic': ks_stat, 'p_value': ks_p},
        'anderson': anderson_result
    }


def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the mean
    
    Parameters:
    -----------
    data : array-like
        The data
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    --------
    tuple : (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean, mean - margin, mean + margin


def compare_groups(group1, group2, test_type='independent', alpha=0.05):
    """
    Compare two groups using appropriate statistical test
    
    Parameters:
    -----------
    group1, group2 : array-like
        The two groups to compare
    test_type : str
        'independent' or 'paired'
    alpha : float
        Significance level
    """
    if test_type == 'independent':
        t_stat, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent T-Test"
    elif test_type == 'paired':
        t_stat, p_value = stats.ttest_rel(group1, group2)
        test_name = "Paired T-Test"
    else:
        raise ValueError("test_type must be 'independent' or 'paired'")
    
    print("=" * 50)
    print(f"{test_name}")
    print("=" * 50)
    print(f"Group 1 Mean: {np.mean(group1):.4f} (SD: {np.std(group1, ddof=1):.4f})")
    print(f"Group 2 Mean: {np.mean(group2):.4f} (SD: {np.std(group2, ddof=1):.4f})")
    print(f"Difference:   {np.mean(group1) - np.mean(group2):.4f}")
    print()
    print(f"T-statistic:  {t_stat:.4f}")
    print(f"P-value:      {p_value:.4f}")
    print()
    print(f"Result (α={alpha}):")
    if p_value < alpha:
        print("  ✓ Statistically significant - Reject null hypothesis")
    else:
        print("  ✗ Not statistically significant - Fail to reject null hypothesis")
    print("=" * 50)
    
    return t_stat, p_value


if __name__ == "__main__":
    # Example usage
    print("Statistics Utility Functions Loaded Successfully!")
    print("\nExample usage:")
    print("  from scripts.utils import summarize_data, plot_distribution")
    print("  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    print("  summarize_data(data)")
