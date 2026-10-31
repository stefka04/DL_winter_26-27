import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(0, 10.0, 50)
    noise = rng.normal(0, 1.0, size=50)
    slope = 2.0
    intercept = 1.0
    y = slope * x + intercept + noise
    print(f"Generated {y.size} points for y = {slope}*x + {intercept} + noise")
    print(f"x range:[{round(x.min(), 2)}, {round(x.max(), 2)}]")
    print(f"y range:[{round(y.min(), 2)}, {round(y.max(), 2)}]")

    figure, axis = plt.subplots()
    axis.scatter(x, y)
    axis.set_title("Synthetic linear data with noise")
    axis.set_xlabel("x")
    axis.set_ylabel("y")

    figure.savefig("w01_task11_synthetic_data.png")
    # plt.show()

    return 0
