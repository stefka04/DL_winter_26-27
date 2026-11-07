from dl_lib.perceptron import datasets, functional


def main() -> int:
    print(f"Dataset: {datasets.create_dataset(4)}")
    print(f"initialize_weights(0, 100): {functional.initialize_weights(0, 100, 1)}")
    print(f"initialize_weights(0, 10): {functional.initialize_weights(0, 10, 1)}")

    return 0
