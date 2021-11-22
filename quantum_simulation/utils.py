"""
This module provides common functions for quantum computing.
"""
import math
from typing import Union

import numpy as np
import numpy.typing as npt


SQRT1_2 = 1 / math.sqrt(2)


def log2(p: Union[int, float]) -> int:
    """
    Calculate an integer logarithm with base 2, fail if not exist.
    """
    n = int(math.log2(p))
    assert 2 ** n == p
    return n


def hadamard(p: int, dtype: npt.DTypeLike = float) -> np.ndarray:
    """
    Construct an Hadamard matrix of the size `p` (should be a power of 2).
    """
    n = log2(p)
    assert issubclass(np.dtype(dtype).type, np.inexact)
    if n > 1:
        return np.kron(hadamard(2, dtype), hadamard(p // 2, dtype))
    if n == 1:
        return SQRT1_2 * np.array([[1, 1], [1, -1]], dtype)
    return np.array([[1]], dtype)


def phase_oracle(f: npt.ArrayLike, dtype: npt.DTypeLike = float) -> np.ndarray:
    """
    Encode a binary function into an oracle given its values on the basis
    states.
    """
    assert issubclass(np.dtype(dtype).type, (np.inexact, np.signedinteger))
    assert np.isin(f, [0, 1]).all()  # Check if the given function is binary
    f_array = np.asarray(f, int)
    p = len(f_array)
    o_f = np.zeros((p, p), dtype)
    diagonal = (-np.ones(p, dtype)) ** f_array
    np.fill_diagonal(o_f, diagonal)
    return o_f
