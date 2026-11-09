import pytest

from pcntoolkit.math_functions.warp import *


def test_parseWarpString():
    warp = parseWarpString("WarpLog")
    assert isinstance(warp, WarpLog)
    warp = parseWarpString("WarpCompose(WarpLog, WarpAffine)")
    assert isinstance(warp, WarpCompose)
    assert len(warp.warps) == 2
    assert isinstance(warp.warps[0], WarpLog)
    assert isinstance(warp.warps[1], WarpAffine)
    warp = parseWarpString("WarpCompose(WarpLog, WarpAffine, WarpSinhArcsinh)")
    assert isinstance(warp, WarpCompose)
    assert len(warp.warps) == 3
    assert isinstance(warp.warps[0], WarpLog)
    assert isinstance(warp.warps[1], WarpAffine)
    assert isinstance(warp.warps[2], WarpSinhArcsinh)
    warp = parseWarpString(
        "WarpCompose(WarpLog, WarpAffine, WarpSinhArcsinh, WarpBoxCox)"
    )
    assert isinstance(warp, WarpCompose)
    assert len(warp.warps) == 4
    assert isinstance(warp.warps[0], WarpLog)
    assert isinstance(warp.warps[1], WarpAffine)
    assert isinstance(warp.warps[2], WarpSinhArcsinh)
    assert isinstance(warp.warps[3], WarpBoxCox)


def test_warplog():
    warp = WarpLog()
    x = np.array([1, 2, 3])
    param = None
    assert isinstance(warp, WarpLog)
    assert warp.get_n_params() == 0
    y = np.log(x)
    assert np.allclose(warp.f(x, param), y)
    assert np.allclose(warp.invf(y, param), x)
    assert np.allclose(warp.df(x, param), 1 / x)


@pytest.mark.parametrize("a,b", [(0, 1), (1, 2), (2, 3)])
def test_warpaffine(a, b):
    warp = WarpAffine()
    x = np.array([1, 2, 3])
    param = np.array([a, b])
    assert isinstance(warp, WarpAffine)
    assert warp.get_n_params() == 2
    y = a + np.exp(b) * x
    assert np.allclose(warp.f(x, param), y)
    assert np.allclose(warp.invf(y, param), x)
    assert np.allclose(warp.df(x, param), np.exp(b))


@pytest.mark.parametrize("lmb", [0, 1, 2])
def test_warpboxcox(lmb):
    warp = WarpBoxCox()
    x = np.array([1, 2, 3])
    param = np.array([lmb])
    assert isinstance(warp, WarpBoxCox)
    assert warp.get_n_params() == 1

    def bct(x, lmb):
        if lmb == 0:
            return np.log(x)
        else:
            return (np.sign(x) * np.abs(x) ** lmb - 1) / lmb

    y = bct(x, np.exp(lmb))
    assert np.allclose(warp.f(x, param), y)
    assert np.allclose(warp.invf(y, param), x)
    assert np.allclose(warp.df(x, param), np.sign(x) * np.abs(x) ** (np.exp(lmb) - 1))


@pytest.mark.parametrize("epsilon,delta", [(0, 1), (1, 2), (2, 3)])
def test_warpsinharcsinh(epsilon, delta):
    b = np.exp(delta)
    a = -epsilon * b

    warp = WarpSinhArcsinh()
    x = np.array([1, 2, 3])
    param = np.array([epsilon, delta])
    assert isinstance(warp, WarpSinhArcsinh)
    assert warp.get_n_params() == 2
    y = np.sinh(b * np.arcsinh(x) - a)
    assert np.allclose(warp.f(x, param), y)
    assert np.allclose(warp.invf(y, param), x)
    assert np.allclose(
        warp.df(x, param), b * np.cosh(b * np.arcsinh(x) - a) / np.sqrt(x**2 + 1)
    )


def test_warpcompose():
    warp = WarpCompose([WarpLog(), WarpAffine(), WarpSinhArcsinh(), WarpBoxCox()])
    x = np.array([0.1, 0.2, 0.3])
    param = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert isinstance(warp, WarpCompose)
    assert warp.get_n_params() == 5
    y = warp.f(x, param)
    new_x = warp.invf(y, param)
    assert np.allclose(x, new_x)
    dy = warp.df(x, param)
    assert np.isfinite(dy).all()
    assert (dy > 0).all()
