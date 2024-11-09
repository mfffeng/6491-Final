import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

E = 6
D = 10
r = 0.03
T = 1
sigma_E = 0.60


def merton_equations(vars):
    V, sigma_V = vars
    d1 = (np.log(V / D) + (r + sigma_V**2 / 2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)

    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = V * norm.cdf(d1) * sigma_V - E * sigma_E
    return [eq1, eq2]


def calc_sigma_V(sigma_E):
    def merton_equations(vars):
        V, sigma_V = vars
        d1 = (np.log(V / D) + (r + sigma_V**2 / 2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)

        eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        eq2 = V * norm.cdf(d1) * sigma_V - E * sigma_E
        return [eq1, eq2]

    initial_guess = [E + D, sigma_E]
    V0, sigma_V = fsolve(merton_equations, initial_guess)
    return sigma_V


# Initial guess for V and sigma_V
initial_guess = [E + D, sigma_E]

V0, sigma_V = fsolve(merton_equations, initial_guess)


sigma_E_range = np.linspace(0.2, 1.0, 100)
sigma_V_values = []
for sigma_E in sigma_E_range:
    sigma_V_values.append(calc_sigma_V(sigma_E))

d1 = (np.log(V0 / D) + (r + sigma_V**2 / 2) * T) / (sigma_V * np.sqrt(T))
d2 = d1 - sigma_V * np.sqrt(T)

# Probability of default
prob_default = norm.cdf(-d2)

plt.figure(figsize=(10, 6))
plt.plot(np.array(sigma_E_range) * 100, np.round(np.array(sigma_V_values) * 100, 2))
plt.xlabel("Equity Volatility (\\sigma_E) in %")
plt.ylabel("Asset Volatility (\\sigma_V) in %")
plt.title("Relationship between Equity and Asset Volatility")
plt.grid(True)

print("Results:")
print(f"d1 = {d1:.4f}")
print(f"d2 = {d2:.4f}")
print(f"Initial Asset Value (V_0): ${V0:.2f} million")
print(f"Asset Volatility (\\sigma_V): {sigma_V:.2%}")
print(f"Probability of Default: {prob_default:.2%}")

plt.show()
