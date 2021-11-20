"""
This module provides single quantum register class.
"""
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

from .qubit import Qubit
from .utils import hadamard, phase_oracle


class QuantumRegister:
    """
    A single quantum register.
    """

    _n: int
    _state: np.ndarray

    @property
    def n(self) -> int:
        """
        Number of qubits in the quantum register.
        """
        return self._n

    @property
    def p(self) -> int:
        """
        Number of basis states in the quantum register.
        """
        return 2 ** self._n

    @property
    def state(self) -> np.ndarray:
        """
        Get the the current state.
        """
        return self._state.copy()

    @property
    def real(self) -> np.ndarray:
        """
        Get the real part of the current state.
        """
        return self.state.real

    @property
    def imag(self) -> np.ndarray:
        """
        Get the imaginary part of the current state.
        """
        return self.state.imag

    def __init__(self, qubits: Sequence[Union[str, int, Qubit]]) -> None:
        assert len(qubits) > 0
        self._n = len(qubits)
        if all(isinstance(q, Qubit) for q in qubits):
            self._state = np.ones(1, complex)
            for q in reversed(qubits):
                self._state = np.kron(q, self._state)
        else:
            if all(isinstance(q, int) for q in qubits):
                assert set(qubits).issubset({0, 1})
                qubits = list(map(str, qubits))
            assert all(isinstance(q, str) for q in qubits)
            assert set(qubits).issubset({"0", "1"})
            self._state = np.zeros(self.p, complex)
            self._state[int("".join(qubits), 2)] = 1
        assert len(self._state) == self.p

    def __repr__(self) -> str:
        return "(" + ", ".join(f"{x:.2f}" for x in self.real) + ")"

    def __array__(self) -> np.ndarray:
        return self.state

    def apply(self, gate: npt.ArrayLike) -> "QuantumRegister":
        """
        Apply a gate to the current state of the quantum register.
        """
        gate_array = np.asarray(gate, float)
        assert gate_array.shape == (self.p, self.p)
        self._state = gate_array @ self._state
        return self

    def h(self) -> "QuantumRegister":
        """
        Apply Hadamard gate to the current state of the quantum register.
        """
        return self.apply(hadamard(self.p, float))

    def r(self) -> "QuantumRegister":
        """
        Apply diffusion gate to the current state of the quantum register.
        """
        r_gate = 2 / self.p * np.ones((self.p, self.p), float) - np.eye(
            self.p, dtype=float
        )
        return self.apply(r_gate)

    def o_f(self, f: npt.ArrayLike) -> "QuantumRegister":
        """
        Apply a phase oracle given its values on the basis states to the
        current state of the quantum register.
        """
        return self.apply(phase_oracle(f, float))


QRegister = QuantumRegister
