import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    rng = np.random.default_rng(seed=42)

    uniform_samples = rng.uniform(size=10000)
    uniform_axis = axes.flat[0]
    uniform_axis.hist(uniform_samples, bins=30)
    uniform_axis.grid(True)
    uniform_axis.set_xlabel("value")
    uniform_axis.set_ylabel("count")
    uniform_axis.set_title("Uniform")
    print(
        f"Uniform: mean={uniform_samples.mean():.4f}, std={uniform_samples.std():.4f}, min={uniform_samples.min():.4f}, max={uniform_samples.max():.4f}"
    )

    normal_samples = rng.normal(size=10000)
    normal_axis = axes.flat[1]
    normal_axis.hist(normal_samples, bins=30)
    normal_axis.grid(True)
    normal_axis.set_xlabel("value")
    normal_axis.set_ylabel("count")
    normal_axis.set_title("Normal")
    print(
        f"Normal: mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}, min={normal_samples.min():.4f}, max={normal_samples.max():.4f}"
    )

    exponential_samples = rng.exponential(size=10000)
    exponential_axis = axes.flat[2]
    exponential_axis.hist(exponential_samples, bins=30)
    exponential_axis.grid(True)
    exponential_axis.set_xlabel("value")
    exponential_axis.set_ylabel("count")
    exponential_axis.set_title("Exponential")
    print(
        f"Exponential: mean={exponential_samples.mean():.4f}, std={exponential_samples.std():.4f}, min={exponential_samples.min():.4f}, max={exponential_samples.max():.4f}"
    )

    figure.tight_layout()
    plt.show()

    # figure.savefig("w01_task13_uniform_normal_exponential.png")
    return 0
