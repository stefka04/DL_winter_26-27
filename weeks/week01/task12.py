import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    x = np.linspace(0, 6)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axis = axes.flat[0]
    axis.plot(x, np.sin(x), label="sin(x)")
    axis.plot(x, np.cos(x), label="cos(x)")
    axis.grid(True)
    axis.legend()
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Line plot")

    rng = np.random.default_rng(seed=42)
    axisHist = axes.flat[1]
    axisHist.hist(rng.normal(size=1000), bins=30)
    axisHist.grid(True)
    axisHist.set_xlabel("value")
    axisHist.set_ylabel("count")
    axisHist.set_title("Histogram")

    figure.tight_layout()

    print(f"sin(x) range: [{round(np.sin(x).min(), 4)}, {round(np.sin(x).max(), 4)}]")
    print("Histogram sample count: 1000")
    plt.show()

    # figure.savefig("w01_task12_line_plot_and_hist.png")
    return 0
