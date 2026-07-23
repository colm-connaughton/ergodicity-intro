import numpy as np
import matplotlib.pyplot as plt


def stationary_density_operator_iteration(
    a,
    n_grid=4001,
    n_iter=100,
    tol=1e-12,
):
    """
    Compute the stationary density for

        X_{n+1} = a X_n + ξ_n,
        ξ_n ~ Uniform[-1,1]

    by iterating the density operator.
    """

    if not (0 < a < 1):
        raise ValueError("Require 0 < a < 1")

    # support of stationary distribution
    L = 1.0 / (1.0 - a)

    x = np.linspace(-L, L, n_grid)
    dx = x[1] - x[0]

    # Initial guess: uniform density
    p = np.ones_like(x) / (2 * L)

    snapshots = {0: p.copy()}
    errors = []

    for iteration in range(1, n_iter + 1):

        #
        # Compute cumulative integral
        #
        F = np.zeros_like(x)
        F[1:] = np.cumsum(
            0.5 * (p[:-1] + p[1:]) * dx
        )

        #
        # Evaluate operator
        #
        lower = (x - 1) / a
        upper = (x + 1) / a

        Flow = np.interp(lower, x, F,
                         left=0.0,
                         right=F[-1])

        Fhigh = np.interp(upper, x, F,
                          left=0.0,
                          right=F[-1])

        p_new = 0.5 * (Fhigh - Flow)

        #
        # Remove tiny numerical negatives
        #
        p_new = np.maximum(p_new, 0)

        #
        # Renormalise
        #
        p_new /= np.trapezoid(p_new, x)

        #
        # Convergence monitor
        #
        err = np.trapezoid(np.abs(p_new - p), x)
        errors.append(err)

        p = p_new

        if iteration in [1,2,5,10,25,50]:
            snapshots[iteration] = p.copy()

        if err < tol:
            break

    return x, p, snapshots, np.array(errors)
