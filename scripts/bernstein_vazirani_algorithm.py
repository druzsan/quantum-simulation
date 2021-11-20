"""
This script shows how Bernstein-Vazirani algorithm
(https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm) works.
"""

import random
from typing import List

from quantum_simulation.algorithms import bernstein_vazirani_algorithm


def encode(key: List[int]) -> List[int]:
    """
    Encode all basis states with the key s, so that
    f = s * x = (s1 * x1 + s2 * x2 + ... + sn * xn) mod 2.
    """
    n = len(key)
    value = int("".join(map(str, key)), 2)
    return [bin(value & x).count("1") % 2 for x in range(2 ** n)]


def main() -> None:
    print("Bernstein-Vazirani algorithm.")
    print("\nEncoded functions.")
    for n in range(1, 8):
        key = random.choices([0, 1], k=n)
        print(f"\nKey to find is: {key}")
        f = encode(key)
        found_key = bernstein_vazirani_algorithm(f, verbose=2)
        assert found_key == key

    print("\nRandom functions.")
    for n in range(3, 5):
        print()
        f = random.choices([0, 1], k=2 ** n)
        found_key = bernstein_vazirani_algorithm(f, verbose=2)
        print(found_key)


if __name__ == "__main__":
    main()
