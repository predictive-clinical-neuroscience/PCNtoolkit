"""When does nbinom.ppf actually return inf? Is it reachable in real use?"""
import numpy as np
from scipy.stats import nbinom, norm

print("=== At which Z does the guard actually fire? ===")
print("(centiles the toolkit computes by default: 0.05 .. 0.95, i.e. |Z| < 2)\n")

for psi, mu, alpha in [(0.7, 3.0, 0.4), (0.5, 20.0, 1.0), (0.9, 1.5, 5.0), (0.3, 8.0, 2.0)]:
    n, p = alpha, alpha / (alpha + mu)
    print(f"psi={psi} mu={mu} alpha={alpha}")
    for z in [1.96, 2.58, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.2, 8.3, 9.0, 10.0]:
        U = norm.cdf(z)
        U_nb = np.clip((U - (1 - psi)) / psi, 0.0, 1.0)
        q = nbinom.ppf(U_nb, n, p)
        flag = "  <-- inf (guard fires)" if not np.isfinite(q) else ""
        print(f"   Z={z:<5} U={U:.17f} U_nb={U_nb:.17f} -> y={q}{flag}")
    print()

print("=== Why: U_nb hits EXACTLY 1.0 due to float64 precision ===")
print("norm.cdf(z) becomes exactly 1.0 at z >=", end=" ")
z = 8.0
while norm.cdf(z) < 1.0:
    z += 0.01
print(f"{z:.2f}")
print("norm.cdf(8.29) =", repr(norm.cdf(8.29)))
print("norm.cdf(8.30) =", repr(norm.cdf(8.30)))
print()
print("So the guard only fires when Z is so large that norm.cdf saturates to 1.0.")
print("That is a probability of 1 - 1e-16, i.e. beyond any real centile.")

print()
print("=== What do the OTHER likelihoods do at the same extreme? ===")
from scipy.stats import beta as beta_dist
print("Beta.backward (stats.beta.ppf) at Z=10:", beta_dist.ppf(norm.cdf(10.0), 2.0, 3.0))
print("Beta.backward at Z=40:", beta_dist.ppf(norm.cdf(40.0), 2.0, 3.0))
print("Normal.backward at Z=40: ", 40.0 * 1.0 + 0.0)
print("-> they return the distribution's upper bound / a finite number, no error")

print()
print("=== Largest count actually reachable before saturation ===")
for psi, mu, alpha in [(0.7, 3.0, 0.4), (0.5, 20.0, 1.0)]:
    n, p = alpha, alpha / (alpha + mu)
    U_nb = np.nextafter(1.0, 0.0)   # largest float64 below 1
    print(f"  psi={psi} mu={mu} alpha={alpha}: max finite count = {nbinom.ppf(U_nb, n, p):.0f}")
