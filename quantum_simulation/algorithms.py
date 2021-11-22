"""
This module provides a set of quantum computing algorithms.
"""
import math
from typing import List, Union

import numpy as np
import numpy.typing as npt

from .quantum_register import QRegister
from .qubit import zero
from .utils import log2, phase_oracle


def deutsch_jozsa_algorithm(f: npt.ArrayLike, verbose: Union[bool, int] = False) -> str:
    """
    Deutsch-Jozsa algorithm
    (https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm).

    Determine if function is constant 0, constant 1 or balanced
    (the half of its values is 0 and the other half is 1).
    """
    if verbose > 0:
        print(f"Function values on the basis states: {f}")
    p = len(f)
    n = log2(p)
    o_f = phase_oracle(f)
    qubits = [zero] * n
    if verbose > 1:
        print(f"Initial qubits: {qubits}")
    qregister = QRegister(qubits)
    if verbose > 1:
        print(f"Initial quantum register: {qregister}")

    qregister.h()
    if verbose > 1:
        print(f"Quantum register after Hadamard gate: {qregister}")
    qregister.apply(o_f)
    if verbose > 1:
        print(f"Quantum register after phase oracle gate: {qregister}")
    qregister.h()
    if verbose > 1:
        print(f"Quantum register after the second Hadamard gate: {qregister}")

    flag = qregister.real[0]
    if np.allclose(flag, 1):
        found_type = "constant 0"
    elif np.allclose(flag, -1):
        found_type = "constant 1"
    elif np.allclose(flag, 0):
        found_type = "balanced"
    else:
        found_type = "other"
    if verbose > 0:
        print(f"Found type: {found_type}")
    return found_type


def grover_algorithm(f: npt.ArrayLike, verbose: Union[bool, int] = False) -> List[int]:
    """
    Grover's algorithm (https://en.wikipedia.org/wiki/Grover%27s_algorithm).

    Find the only index at which function equals to 1.
    If multiple 1's received, answer can be wrong.
    """
    if verbose > 0:
        print(f"Function values on the basis states: {f}")
    p = len(f)
    n = log2(p)
    o_f = phase_oracle(f)
    k = math.floor(np.pi * math.sqrt(2 ** n) / 4)
    if verbose > 1:
        print(f"Number of iterations: {k}")
    qubits = [zero] * n
    if verbose > 1:
        print(f"Initial qubits: {qubits}")
    qregister = QRegister(qubits)
    if verbose > 1:
        print(f"Initial quantum register: {qregister}")

    qregister.h()
    if verbose > 1:
        print(f"Quantum register after Hadamard gate: {qregister}")
    for i in range(k):
        qregister.apply(o_f).r()
        if verbose > 1:
            print(f"Quantum register after {i + 1}th Grover operator: {qregister}")

    found_index = list(map(int, format(np.square(qregister.real).argmax(), f"0{n}b")))
    if verbose > 0:
        print(f"Found index: {found_index}")
    return found_index


def bernstein_vazirani_algorithm(
    f: npt.ArrayLike, verbose: Union[bool, int] = False
) -> List[int]:
    """
    Bernstein-Vazirani algorithm
    (https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm).

    Find the secret key of the function, so that
    f = s * x = (s1 * x1 + s2 * x2 + ... + sn * xn) mod 2.
    """
    if verbose > 0:
        print(f"Function values on the basis states: {f}")
    p = len(f)
    n = log2(p)
    o_f = phase_oracle(f)
    qubits = [zero] * n
    if verbose > 1:
        print(f"Initial qubits: {qubits}")
    qregister = QRegister(qubits)
    if verbose > 1:
        print(f"Initial quantum register: {qregister}")

    qregister.h()
    if verbose > 1:
        print(f"Quantum register after Hadamard gate: {qregister}")
    qregister.apply(o_f)
    if verbose > 1:
        print(f"Quantum register after phase oracle gate: {qregister}")
    qregister.h()
    if verbose > 1:
        print(f"Quantum register after the second Hadamard gate: {qregister}")

    found_key = list(map(int, format(qregister.real.argmax(), f"0{n}b")))
    if verbose > 0:
        print(f"Found key: {found_key}")
    return found_key
