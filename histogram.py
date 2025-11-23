import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
n_simulations = 10000  # Number of training sets to generate
n_samples = 200  # Size of each training set (n)
delta = 0.05  # Confidence parameter (1 - delta = 0.95)
h = 1  # VC dimension derived in previous steps

# --- Simulation ---
# Store the risk differences: R(f_a) - R_hat(f_a)
risk_differences = []

print(f"Running {n_simulations} simulations...")

for _ in range(n_simulations):
    # 1. Generate training set S: n points from Uniform[-1, 1]
    S_x = np.random.uniform(-1, 1, n_samples)

    # 2. Identify the ERM parameter 'a'
    # a = min(x) such that x is positive (label y=1).
    # If no positive x exists, a = 1.
    positive_examples = S_x[S_x >= 0]

    if len(positive_examples) > 0:
        a = np.min(positive_examples)
    else:
        a = 1.0

    # 3. Calculate Risks
    # True Risk R(f_a): The probability mass in [0, a). For Uniform[-1, 1], this is a/2.
    true_risk = a / 2.0

    # Empirical Risk R_hat(f_a): By definition of this ERM on this dataset, it is 0.
    empirical_risk = 0.0

    # Store the difference
    risk_differences.append(true_risk - empirical_risk)

risk_differences = np.array(risk_differences)

# --- Analysis ---

# 1. Calculate the Empirical 95% Quantile
# We want the value such that 95% of the risk differences are below it.
quantile_95 = np.percentile(risk_differences, 95)

# 2. Calculate the Theoretical PAC Bound
# Formula: 2 * sqrt( (2/n) * ( h*ln(n) + h*ln(2e) + ln(2/delta) ) )
term1 = h * np.log(n_samples)
term2 = h * np.log(2 * np.e)
term3 = np.log(2 / delta)

# Note: In the problem statement, the "h" in the denominator of the log term
# technically vanishes or is 1, simplifying to just log(2e) or similar.
# Given h=1, the standard VC bound structure applies.
complexity_term = term1 + term2 + term3
bound_val = 2 * np.sqrt((2 / n_samples) * complexity_term)

print("-" * 30)
print(f"Empirical 95% Quantile: {quantile_95:.5f}")
print(f"Theoretical PAC Bound:  {bound_val:.5f}")
print("-" * 30)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Histogram of the risk differences
plt.hist(risk_differences, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Risk (R - R_hat)')

# Vertical line for Empirical 95% Quantile
plt.axvline(quantile_95, color='red', linestyle='dashed', linewidth=2, label=f'95% Quantile ({quantile_95:.4f})')

# Vertical line for Theoretical Bound
plt.axvline(bound_val, color='green', linestyle='dashed', linewidth=2, label=f'PAC Bound ({bound_val:.4f})')

# Labels and formatting
plt.xlabel(r'Generalization Error gap: $R(f_a) - \hat{R}_S(f_a)$')
plt.ylabel('Frequency')
plt.title(f'Distribution of Generalization Error over {n_simulations} Simulations (n={n_samples})')
plt.legend()
plt.grid(True, alpha=0.3)

# Show plot
plt.show()