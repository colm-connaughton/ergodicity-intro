from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt



def load_stationary_density_function():
    module_path = Path(__file__).parent / "discrete-OU.py"
    spec = importlib.util.spec_from_file_location("discrete_ou", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stationary_density_operator_iteration


def main():
    stationary_density_operator_iteration = load_stationary_density_function()

    a = 0.25
    x, p, snapshots, errors = stationary_density_operator_iteration(a=a)

    print(f"Computed stationary density for a={a}")
    print(f"Grid points: {len(x)}")
    print(f"Final L1 iteration error: {errors[-1] if len(errors) else 0.0:.3e}")
    print(f"Stored snapshots at iterations: {sorted(snapshots.keys())}")

    plt.figure(figsize=(8,5))
    plt.plot(x, p, lw=2)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(f"Stationary density (a={a})")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
