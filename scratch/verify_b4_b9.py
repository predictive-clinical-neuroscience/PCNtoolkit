"""Verify B4-B9 fixes."""
import numpy as np
from scipy.stats import norm
from pcntoolkit.math_functions.likelihood import ZeroInflatedNegativeBinomialLikelihood as ZINB

lik = ZINB.__new__(ZINB)
ok = True

def check(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")

mu, alpha, psi = 3.0, 0.4, 0.7
M = lambda v, n=1: np.full(n, v)

print("=== B5: extreme Z no longer wraps to a small/negative count ===")
prev = -1
mono = True
for z in [-10., -5., 0., 3., 5., 8., 10., 20., 38., np.inf]:
    y = int(lik.backward(M(mu), M(alpha), M(psi), Z=M(z))[0])
    print(f"    Z={z:<6} -> Y={y}")
    if y < prev:
        mono = False
    prev = y
check("backward monotonic and non-negative at extreme Z", mono and prev > 0)

print()
print("=== B4: no forced Y>=1 (a zero above F(0) stays possible / low centiles are 0) ===")
lo = int(lik.backward(M(mu), M(alpha), M(psi), Z=M(norm.ppf(0.05)))[0])
check("5th centile is 0, not forced to 1", lo == 0, f"got {lo}")

print()
print("=== B6: forward is seedable / reproducible ===")
Y = np.array([0, 1, 2, 5, 10])
p = (M(mu, 5), M(alpha, 5), M(psi, 5))
z1 = lik.forward(*p, Y=Y, rng=np.random.default_rng(42))
z2 = lik.forward(*p, Y=Y, rng=np.random.default_rng(42))
z3 = lik.forward(*p, Y=Y, rng=np.random.default_rng(7))
check("same seed -> identical Z", np.allclose(z1, z2))
check("different seed -> different Z", not np.allclose(z1, z3))
zA = lik.forward(*p, Y=Y)
zB = lik.forward(*p, Y=Y)
check("no rng passed -> still stochastic (unchanged default)", not np.allclose(zA, zB))

print()
print("=== B7: non-count Y is rejected instead of silently floored ===")
for bad, label in [(np.array([-0.83, 0.12, 1.44]), "scaled/continuous"),
                   (np.array([-1.0, 2.0]), "negative"),
                   (np.array([1.5, 2.0]), "non-integer")]:
    try:
        lik.forward(M(mu, bad.size), M(alpha, bad.size), M(psi, bad.size), Y=bad)
        check(f"rejects {label} Y", False, "no error raised")
    except ValueError as e:
        check(f"rejects {label} Y", True, f"-> {str(e)[:60]}...")

for bad, label in [(np.array([np.nan, 1.0]), "NaN"), (np.array([np.inf, 1.0]), "inf")]:
    try:
        lik.forward(M(mu, 2), M(alpha, 2), M(psi, 2), Y=bad)
        check(f"rejects {label} Y", False, "no error raised")
    except ValueError as e:
        check(f"rejects {label} Y", True, f"-> {str(e)[:50]}...")

# valid integer-valued floats must still be accepted (Y arrives as float from xarray)
try:
    lik.forward(M(mu, 3), M(alpha, 3), M(psi, 3), Y=np.array([0.0, 2.0, 7.0]))
    check("accepts integer-valued floats", True)
except ValueError as e:
    check("accepts integer-valued floats", False, str(e)[:60])

print()
print("=== B8: get_var_names present and consistent with siblings ===")
check("get_var_names defined", hasattr(lik, "get_var_names"))
check("names match param order", lik.get_var_names() == ["mu_samples", "alpha_samples", "psi_samples"],
      str(lik.get_var_names()))

print()
print("=== B9: no debug prints remain ===")
import inspect
src = inspect.getsource(ZINB)
check("no print() in class", "print(" not in src)

print()
print("=== regression: math from B1-B3 still exact ===")
import pymc as pm
from scipy.stats import kstest
s = pm.draw(pm.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha), 100000, random_seed=3).astype(int)
N = s.size
Z = lik.forward(M(mu, N), M(alpha, N), M(psi, N), Y=s, rng=np.random.default_rng(0))
back = lik.backward(M(mu, N), M(alpha, N), M(psi, N), Z=Z)
check("round trip still 100% exact", (back == s).mean() == 1.0, f"{(back==s).mean():.5f}")
check("forward still N(0,1)", kstest(Z, "norm").pvalue > 0.01, f"KS p={kstest(Z,'norm').pvalue:.3f}")

print()
print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
