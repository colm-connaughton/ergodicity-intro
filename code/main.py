from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
import numpy as np

from discrete_ou import simulate_discrete_ou, gaussian_noise, plot_trajectory_and_histogram, bernoulli_noise, plot_stationary_histogram




def load_stationary_density_function():
    module_path = Path(__file__).parent / "discrete_ou.py"
    spec = importlib.util.spec_from_file_location("discrete_ou", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stationary_density_operator_iteration


def main():

    # stationary_density_operator_iteration = load_stationary_density_function()

    # a = 0.25
    # x, p, snapshots, errors = stationary_density_operator_iteration(a=a)

    # print(f"Computed stationary density for a={a}")
    # print(f"Grid points: {len(x)}")
    # print(f"Final L1 iteration error: {errors[-1] if len(errors) else 0.0:.3e}")
    # print(f"Stored snapshots at iterations: {sorted(snapshots.keys())}")

    # plt.figure(figsize=(8,5))
    # plt.plot(x, p, lw=2)
    # plt.xlabel("x")
    # plt.ylabel("density")
    # plt.title(f"Stationary density (a={a})")
    # plt.grid(True)
    # plt.show()

    # a = 0.75

    # samples = simulate_discrete_ou(
    # a=a,
    # noise_generator=gaussian_noise(standard_deviation=1.0),
    # n_samples=200_000,
    # burn_in=5_000,
    # seed=12345,
    # )

    # plot_trajectory_and_histogram(
    #     samples=samples,
    #     a=a,
    #     noise_name="Gaussian",
    #     bins=150,
    # )

    a = 0.5*(np.sqrt(5.0)-1.0)  + 0.01# Golden ratio conjugate
    #a=0.9
    a=0.575

    samples = simulate_discrete_ou(
    a=a,
    noise_generator=bernoulli_noise(amplitude=1.0),
    n_samples=50000000,
    burn_in=5_000,
    seed=12345,
)

    # For symmetric ±1 noise, the support is contained in
    # [-1/(1-a), 1/(1-a)].
    support_bound = 1.0 / (1.0 - abs(a))

    fig, ax = plot_stationary_histogram(
        samples=samples,
        a=a,
        noise_name="Bernoulli ±1",
        bins=2500,
        x_limits=(-support_bound, support_bound),
    )

    filename = f"stationary_histogram_a_{a:.3f}_bernoulli.png"
    # Save the figure with a high resolution (300 dpi)
    fig.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
