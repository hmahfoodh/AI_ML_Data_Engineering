from scipy.stats import ttest_ind

# Sample data
group1 = [1, 2, 3, 4, 5]
group2 = [1, 2, 3, 9, 6]

# Perform a t-test
t_statistic, p_value = ttest_ind(group1, group2)
print("T-statistic:", t_statistic)
print("P-value:", p_value)