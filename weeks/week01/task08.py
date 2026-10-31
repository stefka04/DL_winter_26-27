import time

import numpy as np


def main() -> int:
    rng = np.random.default_rng(seed=42)
    arr = rng.uniform(-1000, 1000, 1000000)
    print(f"Array length: {arr.size}")

    sum = 0
    l_start_time = time.time()
    for num in arr:
        sum += num
    l_elapsed_time = time.time() - l_start_time

    print(f"Loop sum: {round(sum, 4)} ({round(l_elapsed_time, 4)}s)")

    sum = 0
    v_start_time = time.time()
    sum = arr.sum()
    v_elapsed_time = time.time() - v_start_time
    print(f"Vectorized sum: {round(sum, 4)} ({round(v_elapsed_time, 4)}s)")

    v_speed_up = l_elapsed_time / v_elapsed_time
    print(f"Vectorized speedup: {round(v_speed_up, 4)}x\n")

    print(f"Vectorized mean: {round(arr.mean(), 4)}")
    print(f"Vectorized std: {round(arr.std(), 4)}")
    print(f"Vectorized min/max: {round(arr.min(), 4)} / {round(arr.max(), 4)}")

    return 0
