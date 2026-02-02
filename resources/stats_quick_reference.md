# Statistics Quick Reference Guide

## 📊 Descriptive Statistics

### Central Tendency
```python
import numpy as np
import pandas as pd

data = [1, 2, 3, 4, 5]

# Mean
np.mean(data)
pd.Series(data).mean()

# Median
np.median(data)
pd.Series(data).median()

# Mode
from scipy import stats
stats.mode(data)
pd.Series(data).mode()
```

### Dispersion
```python
# Variance
np.var(data, ddof=1)  # Sample variance
pd.Series(data).var()

# Standard Deviation
np.std(data, ddof=1)  # Sample std
pd.Series(data).std()

# Range
np.ptp(data)  # Peak to peak
max(data) - min(data)

# Interquartile Range (IQR)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
```

### Percentiles & Quantiles
```python
# Percentiles
np.percentile(data, [25, 50, 75])

# Quantiles
pd.Series(data).quantile([0.25, 0.5, 0.75])
```

## 📈 Probability Distributions

### Normal Distribution
```python
from scipy import stats

# Create normal distribution
mean, std = 100, 15
normal_dist = stats.norm(loc=mean, scale=std)

# PDF - Probability Density Function
normal_dist.pdf(100)

# CDF - Cumulative Distribution Function
normal_dist.cdf(100)

# Generate random samples
samples = normal_dist.rvs(size=1000)

# Or using numpy
samples = np.random.normal(mean, std, size=1000)
```

### Other Common Distributions
```python
# Binomial
stats.binom(n=10, p=0.5).pmf(5)

# Poisson
stats.poisson(mu=3).pmf(2)

# Exponential
stats.expon(scale=1/0.5).pdf(2)

# t-distribution
stats.t(df=10).pdf(1.5)

# Chi-square
stats.chi2(df=5).pdf(3)

# F-distribution
stats.f(dfn=5, dfd=10).pdf(2)
```

## 🔬 Hypothesis Testing

### T-Tests
```python
from scipy import stats

# One-sample t-test
sample = [2.3, 2.5, 2.7, 2.9, 3.1]
t_stat, p_value = stats.ttest_1samp(sample, popmean=2.5)

# Two-sample independent t-test
sample1 = [2.3, 2.5, 2.7, 2.9, 3.1]
sample2 = [2.1, 2.4, 2.6, 2.8, 3.0]
t_stat, p_value = stats.ttest_ind(sample1, sample2)

# Paired t-test
before = [100, 102, 98, 105, 103]
after = [102, 105, 100, 108, 106]
t_stat, p_value = stats.ttest_rel(before, after)
```

### Chi-Square Tests
```python
# Chi-square goodness of fit
observed = [10, 15, 20, 25]
expected = [15, 15, 20, 20]
chi2_stat, p_value = stats.chisquare(observed, expected)

# Chi-square test of independence
from scipy.stats import chi2_contingency
contingency_table = [[10, 15], [20, 25]]
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

### ANOVA
```python
# One-way ANOVA
group1 = [23, 25, 27, 29]
group2 = [24, 26, 28, 30]
group3 = [22, 24, 26, 28]
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

## 📉 Correlation & Regression

### Correlation
```python
import numpy as np
from scipy import stats

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Pearson correlation
corr, p_value = stats.pearsonr(x, y)

# Spearman correlation (non-parametric)
corr, p_value = stats.spearmanr(x, y)

# Kendall's tau
corr, p_value = stats.kendalltau(x, y)
```

### Linear Regression
```python
from scipy import stats
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Simple linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Predictions
predictions = slope * x + intercept

# R-squared
r_squared = r_value ** 2
```

### Multiple Regression (statsmodels)
```python
import statsmodels.api as sm
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'y': [1, 2, 3, 4, 5],
    'x1': [2, 3, 4, 5, 6],
    'x2': [1, 2, 2, 3, 4]
})

# Add constant for intercept
X = sm.add_constant(df[['x1', 'x2']])
y = df['y']

# Fit model
model = sm.OLS(y, X).fit()

# Summary
print(model.summary())

# Predictions
predictions = model.predict(X)
```

## 📊 Confidence Intervals

### Mean Confidence Interval
```python
from scipy import stats
import numpy as np

data = [23, 25, 27, 29, 31, 33, 35]
confidence = 0.95

mean = np.mean(data)
std_err = stats.sem(data)  # Standard error
margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

ci_lower = mean - margin
ci_upper = mean + margin
```

### Proportion Confidence Interval
```python
from statsmodels.stats.proportion import proportion_confint

successes = 48
total = 100
confidence = 0.95

ci_lower, ci_upper = proportion_confint(successes, total, alpha=1-confidence)
```

## 📐 Statistical Power Analysis
```python
from statsmodels.stats.power import ttest_power

# Calculate required sample size
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.8

required_power = ttest_power(effect_size, nobs=None, alpha=alpha, alternative='two-sided')
```

## 🎲 Random Sampling

### Basic Sampling
```python
import numpy as np

population = list(range(1, 101))

# Simple random sample
sample = np.random.choice(population, size=10, replace=False)

# Sample with replacement
sample_with_replacement = np.random.choice(population, size=10, replace=True)

# Stratified sampling (using pandas)
import pandas as pd
df = pd.DataFrame({'group': ['A']*50 + ['B']*50, 'value': range(100)})
stratified = df.groupby('group', group_keys=False).apply(lambda x: x.sample(5))
```

## 📊 Data Visualization Templates

### Histograms
```python
import matplotlib.pyplot as plt

plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

### Box Plot
```python
import seaborn as sns

sns.boxplot(data=df, y='value', x='group')
plt.title('Box Plot')
plt.show()
```

### Scatter Plot with Regression
```python
sns.regplot(x='x', y='y', data=df)
plt.title('Scatter Plot with Regression Line')
plt.show()
```

### Q-Q Plot (Check Normality)
```python
from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

## 🔍 Checking Assumptions

### Normality Tests
```python
from scipy import stats

# Shapiro-Wilk test
statistic, p_value = stats.shapiro(data)

# Kolmogorov-Smirnov test
statistic, p_value = stats.kstest(data, 'norm')

# Anderson-Darling test
result = stats.anderson(data)
```

### Homogeneity of Variance
```python
# Levene's test
from scipy.stats import levene
statistic, p_value = levene(group1, group2, group3)

# Bartlett's test (for normal data)
from scipy.stats import bartlett
statistic, p_value = bartlett(group1, group2, group3)
```

## 💡 Tips

1. **Always visualize your data first** before running tests
2. **Check assumptions** before applying parametric tests
3. **Use appropriate significance level** (usually α = 0.05)
4. **Consider effect size** not just p-values
5. **Use non-parametric tests** when assumptions are violated
6. **Adjust for multiple comparisons** when doing many tests

## 🔗 Common Workflows

### Complete Analysis Template
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Exploratory analysis
print(df.describe())
print(df.info())

# 3. Visualize
sns.histplot(df['variable'])
plt.show()

# 4. Check normality
stats.shapiro(df['variable'])

# 5. Statistical test
t_stat, p_value = stats.ttest_ind(df['group1'], df['group2'])

# 6. Interpret results
if p_value < 0.05:
    print("Statistically significant")
else:
    print("Not statistically significant")
```
