"""
This script shows how Deutsch-Jozsa algorithm
(https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm) works.
"""

import random

from quantum_simulation.algorithms import deutsch_jozsa_algorithm


def main() -> None:
    print("Deutsch-Jozsa algorithm.")
    print("\nConstant 0 functions.")
    for n in range(1, 8, 2):
        print()
        f = [0] * 2 ** n
        found_type = deutsch_jozsa_algorithm(f, verbose=2)
        assert found_type == "constant 0"

    print("\nConstant 1 functions.")
    for n in range(1, 8, 2):
        print()
        f = [1] * 2 ** n
        found_type = deutsch_jozsa_algorithm(f, verbose=2)
        assert found_type == "constant 1"

    print("\nBalanced functions.")
    for n in range(1, 8, 2):
        print()
        f = [0, 1] * 2 ** (n - 1)
        random.shuffle(f)
        found_type = deutsch_jozsa_algorithm(f, verbose=2)
        assert found_type == "balanced"

    print("\nRandom functions.")
    for n in range(1, 8, 2):
        print()
        f = random.choices([0, 1], k=2 ** n)
        found_type = deutsch_jozsa_algorithm(f, verbose=2)
        print(f"Function's type: {found_type}")


if __name__ == "__main__":
    main()
