import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses import functional

matplotlib.use("Agg")


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([4.0, 4.0, 7.5, 9.0])
    mae = np.zeros(shape=(10,))
    mse = np.zeros(shape=(10,))
    rmse = np.zeros(shape=(10,))

    growing_errors = np.arange(1, 11)
    for growing_error in growing_errors:
        y_pred[3] += growing_error
        index = growing_error - 1
        mae[index] = functional.mae(y_true, y_pred)
        mse[index] = functional.mse(y_true, y_pred)
        rmse[index] = functional.rmse(y_true, y_pred)
        y_pred[3] -= growing_error

    print(
        f"Growing error={growing_errors[0]:.1f}: MAE={mae[0]:.4f}, MSE={mse[0]:.4f}, RMSE={rmse[0]:.4f}"
    )
    print(
        f"Growing error={growing_errors[4]:.1f}: MAE={mae[4]:.4f}, MSE={mse[4]:.4f}, RMSE={rmse[4]:.4f}"
    )
    print(
        f"Growing error={growing_errors[9]:.1f}: MAE={mae[9]:.4f}, MSE={mse[9]:.4f}, RMSE={rmse[9]:.4f}"
    )

    figure, axis = plt.subplots()
    axis.plot(growing_errors, mae, label="MAE")
    axis.plot(growing_errors, mse, label="MSE")
    axis.plot(growing_errors, rmse, label="RMSE")

    axis.legend()
    axis.set_title("MAE, MSE, and RMSE as one prediction error grows")
    axis.set_xlabel(
        "the growing error, with three other errors (1., -1., 0.5) held fixed)"
    )
    axis.set_ylabel("loss value")
    axis.grid(True)

    figure.tight_layout()
    # plt.show()

    # figure.savefig("w02_loss_comparison.png")

    return 0
