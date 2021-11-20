"""
This module provides single qubit class.
"""
import math
from typing import Union

import numpy as np

from .utils import SQRT1_2


class Qubit:
    """
    A single qubit.
    """

    _alpha: complex
    _beta: complex

    @property
    def alpha(self) -> complex:
        """
        Probability amplitude of the qubit for being `|0>`.
        """
        return self._alpha

    @property
    def beta(self) -> complex:
        """
        Probability amplitude of the qubit for being `|1>`.
        """
        return self._beta

    def __init__(
        self,
        alpha: Union[int, float, complex],
        beta: Union[int, float, complex],
    ) -> None:
        squared_magnitude = abs(alpha) ** 2 + abs(beta) ** 2
        if squared_magnitude > 0:
            factor = 1 / math.sqrt(squared_magnitude)
            self._alpha = complex(factor * alpha)
            self._beta = complex(factor * beta)
        else:
            self._alpha = complex(SQRT1_2)
            self._beta = complex(SQRT1_2)

    def __repr__(self) -> str:
        return f"({self.alpha}, {self.beta})"

    def __array__(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], complex)


zero = Qubit(1, 0)
one = Qubit(0, 1)
