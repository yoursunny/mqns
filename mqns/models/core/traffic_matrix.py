import numpy as np

from mqns.models.core.state import ATOL
from mqns.utils import rng

type TrafficMatrixInput = np.ndarray | list[list[int]]


class TrafficMatrix:
    _matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    """
    Traffic matrix for N nodes, shape is (N,N).
    ``T[i,j]`` defines the probability that node i initiates a request to node j.
    The matrix should be normalized; ``T[i,i]`` must be zero.
    """

    _flat: np.ndarray[tuple[int], np.dtype[np.float64]]
    """
    Flattened probabilities, shape is (N**2,).
    """

    def __init__(self, input: TrafficMatrixInput, n: int):
        try:
            array = np.array(input, dtype=np.float64)
        except ValueError:
            raise ValueError(f"traffic matrix for {n} nodes should have ({n},{n}) shape")
        if array.shape != (n, n):
            raise ValueError(f"traffic matrix for {n} nodes should have ({n},{n}) shape")
        if not np.all(array >= 0):
            raise ValueError("traffic matrix elements must be nonnegative")
        if not np.all(np.diag(array) == 0):
            raise ValueError("traffic matrix diagonal elements T[i,i] must be exactly zero")
        sum = np.sum(array)
        if sum < ATOL:
            raise ValueError("traffic matrix input is zero")
        array /= sum
        self._matrix = array
        self._flat = array.ravel()

    def __getitem__(self, index: tuple[int, int]) -> float:
        """Retrieve a probability value."""
        return self._matrix[index]

    def sample(self) -> tuple[int, int]:
        """Randomly select a src-dst pair."""
        index = rng.choice(len(self._flat), p=self._flat)
        return divmod(index, self._matrix.shape[1])
