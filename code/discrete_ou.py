import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from numpy.typing import NDArray


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




NoiseGenerator = Callable[
    [np.random.Generator, int],
    NDArray[np.float64],
]


def simulate_discrete_ou(
    a: float,
    noise_generator: NoiseGenerator,
    n_samples: int = 100_000,
    burn_in: int = 5_000,
    x0: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Simulate the discrete OU / AR(1) process

        X[n+1] = a X[n] + xi[n+1]

    and return samples after discarding an initial burn-in period.

    Parameters
    ----------
    a:
        Autoregressive coefficient. Stationarity requires abs(a) < 1.
    noise_generator:
        Function with signature noise_generator(rng, size), returning
        `size` independent noise samples.
    n_samples:
        Number of stationary samples to retain.
    burn_in:
        Number of initial iterations to discard.
    x0:
        Initial condition.
    seed:
        Random seed for reproducibility.
    """
    if not -1.0 < a < 1.0:
        raise ValueError("Stationarity requires abs(a) < 1.")

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")

    rng = np.random.default_rng(seed)
    n_total = burn_in + n_samples

    noise = np.asarray(
        noise_generator(rng, n_total),
        dtype=np.float64,
    )

    if noise.shape != (n_total,):
        raise ValueError(
            "noise_generator must return a one-dimensional array "
            f"with shape ({n_total},), but returned {noise.shape}."
        )

    trajectory = np.empty(n_total + 1, dtype=np.float64)
    trajectory[0] = x0

    for n in range(n_total):
        trajectory[n + 1] = a * trajectory[n] + noise[n]

    return trajectory[burn_in + 1 :]


# ---------------------------------------------------------------------
# Noise distributions
# ---------------------------------------------------------------------

def gaussian_noise(
    standard_deviation: float = 1.0,
    mean: float = 0.0,
) -> NoiseGenerator:
    """Return a Gaussian noise generator."""

    def generate(
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        return rng.normal(
            loc=mean,
            scale=standard_deviation,
            size=size,
        )

    return generate


def uniform_noise(
    half_width: float = 1.0,
) -> NoiseGenerator:
    """Return a generator for Uniform[-half_width, half_width] noise."""

    def generate(
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        return rng.uniform(
            low=-half_width,
            high=half_width,
            size=size,
        )

    return generate


def bernoulli_noise(
    amplitude: float = 1.0,
    probability_plus: float = 0.5,
) -> NoiseGenerator:
    """Return noise taking values -amplitude and +amplitude."""

    if not 0.0 <= probability_plus <= 1.0:
        raise ValueError("probability_plus must lie between 0 and 1.")

    def generate(
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        is_plus = rng.random(size) < probability_plus
        return np.where(is_plus, amplitude, -amplitude)

    return generate


def laplace_noise(
    scale: float = 1.0,
    mean: float = 0.0,
) -> NoiseGenerator:
    """Return a Laplace noise generator."""

    def generate(
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        return rng.laplace(
            loc=mean,
            scale=scale,
            size=size,
        )

    return generate


def cauchy_noise(
    scale: float = 1.0,
    location: float = 0.0,
) -> NoiseGenerator:
    """Return a Cauchy noise generator."""

    def generate(
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        return location + scale * rng.standard_cauchy(size=size)

    return generate


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_stationary_histogram(
    samples: NDArray[np.float64],
    a: float,
    noise_name: str,
    bins: int | str | NDArray[np.float64] = 200,
    density: bool = True,
    x_limits: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of simulated stationary samples."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.hist(
        samples,
        bins=bins,
        density=density,
        alpha=0.75,
        edgecolor="none",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Estimated density" if density else "Count")
    ax.set_title(
        rf"Stationary distribution: $a={a:.5f}$, "
        f"noise={noise_name}"
    )

    if x_limits is not None:
        ax.set_xlim(*x_limits)

    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_trajectory_and_histogram(
    samples: NDArray[np.float64],
    a: float,
    noise_name: str,
    bins: int | str | NDArray[np.float64] = 200,
    n_time_points: int = 1_000,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """
    Plot a short part of the stationary trajectory and its histogram.
    """
    n_time_points = min(n_time_points, len(samples))

    # Separate figures rather than subplots make each visualization easier
    # to resize and inspect independently.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.arange(n_time_points), samples[:n_time_points], linewidth=0.8)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$X_n$")
    ax.set_title(
        rf"Stationary trajectory: $a={a}$, "
        f"noise={noise_name}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    plot_stationary_histogram(
        samples=samples,
        a=a,
        noise_name=noise_name,
        bins=bins,
        x_limits=x_limits,
    )
