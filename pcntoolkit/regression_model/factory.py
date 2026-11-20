from pcntoolkit import (
    BLR,
    BsplineBasisFunction,
    LinearBasisFunction,
    SHASHbLikelihood,
)
from pcntoolkit.math_functions.likelihood import (
    NormalLikelihood,
    get_default_normal_likelihood,
)
from pcntoolkit.math_functions.prior import LinearPrior, Prior, RandomPrior


def linear_BLR():
    # Linear BLR with a fixed effect in the intercept of the mean, and heteroskedasticity
    BLR_model = BLR(
        name="linear_BLR",
        fixed_effect=True,
        heteroskedastic=True,
        basis_function_mean=LinearBasisFunction(basis_column=0),
        basis_function_var=LinearBasisFunction(basis_column=0),
        n_iter=1000,
        tol=1e-3,
        optimizer="l-bfgs-b",
        l_bfgs_b_epsilon=0.1,
        l_bfgs_b_l=0.1,
        l_bfgs_b_norm="l2",
    )
    return BLR_model


def warped_BLR():
    # Warped BLR with a fixed effect in the intercept of the mean, and BSplineheteroskedasticity
    BLR_model = BLR(
        name="warped_BLR",
        fixed_effect=True,
        heteroskedastic=True,
        warp_name="WarpSinhArcSinh",
        warp_reparam=True,
        basis_function_mean=BsplineBasisFunction(basis_column=0, degree=3, nknots=5),
        basis_function_var=BsplineBasisFunction(basis_column=0, degree=3, nknots=5),
        n_iter=1000,
        tol=1e-3,
        optimizer="l-bfgs-b",
        l_bfgs_b_epsilon=0.1,
        l_bfgs_b_l=0.1,
        l_bfgs_b_norm="l2",
    )
    return BLR_model


def Normal_likelihood() -> NormalLikelihood:
    # Normal likelihood with a random intercept of the mean, and heteroskedasticity
    return get_default_normal_likelihood()


def SHASHb1_likelihood() -> SHASHbLikelihood:
    # BSpline regression in mu with random intercept
    mu = LinearPrior(
        slope=Prior(dist_name="Normal", dist_params=(0, 10), dims=("covariates",)),
        intercept=RandomPrior(
            mu=Prior(
                dist_name="Normal",
                dist_params=(0, 3),
            ),
            sigma=Prior(
                dist_name="Normal",
                dist_params=(1.0, 1.0),
                mapping="softplus",
                mapping_params=(0, 3),
            ),
        ),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    # BSpline regression in sigma with fixed intercept
    sigma = LinearPrior(
        slope=Prior(
            dist_name="Normal",
            dist_params=(0, 2),
        ),
        intercept=Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
        ),
        mapping="softplus",
        mapping_params=(0, 3),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    # Fixed values for epsilon and delta
    epsilon = Prior(
        dist_name="Normal",
        dist_params=(0.0, 1.0),
    )
    delta = Prior(
        dist_name="Normal",
        dist_params=(0.0, 2.0),
        mapping="softplus",
        mapping_params=(0, 3, 0.6),
    )
    return SHASHbLikelihood(mu, sigma, epsilon, delta)


def SHASHb2_likelihood() -> SHASHbLikelihood:
    # BSpline regression in mu with random intercept
    mu = LinearPrior(
        slope=Prior(dist_name="Normal", dist_params=(0, 10), dims=("covariates",)),
        intercept=RandomPrior(
            mu=Prior(
                dist_name="Normal",
                dist_params=(0, 3),
            ),
            sigma=Prior(
                dist_name="Normal",
                dist_params=(1.0, 1.0),
                mapping="softplus",
                mapping_params=(0, 3),
            ),
        ),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    # BSpline regression in sigma with fixed intercept
    sigma = LinearPrior(
        slope=Prior(
            dist_name="Normal",
            dist_params=(0, 2),
        ),
        intercept=Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
        ),
        mapping="softplus",
        mapping_params=(0, 3),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    # BSpline regression in epsilon with fixed intercept
    epsilon = LinearPrior(
        slope=Prior(
            dist_name="Normal",
            dist_params=(0, 2),
        ),
        intercept=Prior(
            dist_name="Normal",
            dist_params=(0.0, 1.0),
        ),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    # BSpline regression in delta with fixed intercept
    delta = LinearPrior(
        slope=Prior(
            dist_name="Normal",
            dist_params=(0, 2),
        ),
        intercept=Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
        ),
        mapping="softplus",
        mapping_params=(0, 3, 0.6),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )

    return SHASHbLikelihood(mu, sigma, epsilon, delta)
