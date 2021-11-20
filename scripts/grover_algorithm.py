"""
This script shows how the Grover algorithm
(https://en.wikipedia.org/wiki/Grover%27s_algorithm) works.
"""

import random

import numpy as np

from quantum_simulation.algorithms import grover_algorithm


def main() -> None:
    print("Grover algorithm.")
    print("\nRight input data.")
    for n in range(2, 8):
        index = random.choices([0, 1], k=n)
        print(f"\nIndex to find is: {index}")
        f = np.zeros(2 ** n, int)
        f[int("".join(map(str, index)), 2)] = 1
        found_index = grover_algorithm(f, verbose=2)
        assert found_index == index

    print("\nWrong input data.")
    for n in range(3, 5):
        print()
        f = random.choices([0, 1], k=2 ** n)
        grover_algorithm(f, verbose=2)


if __name__ == "__main__":
    main()
