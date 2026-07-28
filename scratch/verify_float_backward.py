"""Verify backward returning float: correctness, ordering, dtype, downstream."""
import numpy as np
import pymc as pm
import xarray as xr
from scipy.stats import kstest, nbinom, norm

from pcntoolkit.math_functions.likelihood import ZeroInflatedNegativeBinomialLikelihood as ZINB

lik = ZINB.__new__(ZINB)
ok = True

def check(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")

M = lambda v, n=1: np.full(n, float(v))

print("=== 1. dtype is float, values are integer-valued ===")
y = lik.backward(M(3.0, 5), M(0.4, 5), M(0.7, 5), Z=norm.ppf([0.1, 0.3, 0.5, 0.9, 0.99]))
check("returns float dtype", y.dtype == np.float64, str(y.dtype))
finite = y[np.isfinite(y)]
check("values are whole numbers", np.all(finite == np.floor(finite)), str(y))

print()
print("=== 2. THE FIX: ordering preserved past saturation (was all-equal before) ===")
zs = [8.0, 8.2, 8.3, 9.0, 10.0, 20.0, 100.0, np.inf]
ys = [float(lik.backward(M(3.0), M(0.4), M(0.7), Z=M(z))[0]) for z in zs]
for z, v in zip(zs, ys):
    print(f"    Z={z:<7} -> Y={v}")
check("no wraparound to negative", all(v >= 0 for v in ys))
check("saturated region is inf, not a fake finite count", all(np.isinf(v) for v in ys[2:]))
check("non-decreasing", all(ys[i] <= ys[i+1] for i in range(len(ys)-1)))

print()
print("=== 3. B1-B3 math unchanged: exact vs analytic quantile function ===")
def true_ppf(U, mu, alpha, psi):
    grid = np.arange(0, 200000)
    F = (1 - psi) + psi * nbinom.cdf(grid, alpha, alpha / (alpha + mu))
    return float(grid[np.searchsorted(F, U, side="left")])

worst_bad = 0
for psi, mu, alpha in [(0.7, 3.0, 0.4), (0.3, 8.0, 2.0), (0.9, 1.5, 5.0), (0.5, 20.0, 1.0)]:
    bad = 0
    for U in np.arange(0.001, 1.0, 0.001):
        got = float(lik.backward(M(mu), M(alpha), M(psi), Z=M(norm.ppf(U)))[0])
        if got != true_ppf(U, mu, alpha, psi):
            bad += 1
    worst_bad = max(worst_bad, bad)
    print(f"    psi={psi} mu={mu} alpha={alpha}: {bad}/999 disagreements")
check("backward matches analytic ppf everywhere", worst_bad == 0)

print()
print("=== 4. round trip + calibration vs PyMC ===")
for psi, mu, alpha in [(0.7, 3.0, 0.4), (0.5, 20.0, 1.0)]:
    s = pm.draw(pm.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha),
                100000, random_seed=5).astype(int)
    N = s.size
    Z = lik.forward(M(mu, N), M(alpha, N), M(psi, N), Y=s, rng=np.random.default_rng(1))
    back = lik.backward(M(mu, N), M(alpha, N), M(psi, N), Z=Z)
    check(f"round trip exact (psi={psi}, mu={mu})", (back == s).all(), f"{(back==s).mean():.5f}")
    check(f"forward N(0,1) (psi={psi}, mu={mu})", kstest(Z, 'norm').pvalue > 0.01,
          f"KS p={kstest(Z,'norm').pvalue:.3f}")

print()
print("=== 5. downstream: shapes + the .mean(dim='sample') the caller applies ===")
n_obs, n_samp = 40, 100
MU, AL, PS = (np.full((n_obs, n_samp), v) for v in (3.0, 0.4, 0.7))
Zcol = norm.ppf(np.linspace(0.01, 0.99, n_obs))[:, None]
out = lik.backward(MU, AL, PS, Z=Zcol)
check("broadcasts (n,1) Z vs (n,samples) params", out.shape == (n_obs, n_samp), str(out.shape))
da = xr.DataArray(out, dims=["observations", "sample"]).mean(dim="sample")
check("survives xarray mean, no NaN/inf in normal centile range",
      np.all(np.isfinite(da.values)), f"dtype={da.dtype}")

# extreme case: one saturated draw among many
mixed = np.concatenate([norm.ppf(np.full(n_samp - 1, 0.5)), [50.0]])
o2 = lik.backward(np.full(n_samp, 3.0), np.full(n_samp, 0.4), np.full(n_samp, 0.7), Z=mixed)
m2 = xr.DataArray(o2[None, :], dims=["observations", "sample"]).mean(dim="sample")
check("one saturated draw -> inf propagates visibly (not silently capped)",
      np.isinf(m2.values[0]), f"mean={m2.values[0]}")

print()
print("=== 6. constant removed, no int cast left ===")
import inspect
src = inspect.getsource(ZINB)
mod = inspect.getsource(inspect.getmodule(ZINB))
check("_UNBOUNDED_COUNT gone", "_UNBOUNDED_COUNT" not in mod)
check("no .astype(int) in class", ".astype(int)" not in src)

print()
print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
