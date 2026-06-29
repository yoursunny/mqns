import functools
from abc import abstractmethod
from collections.abc import Iterable
from typing import Literal, cast, override

import numpy as np

from mqns.models.core.bell_diagonal import (
    BellDiagonalProbV,
    PauliTransferMat,
    bell_diagonal_probv_to_pauli_transfer_mat,
    make_bell_diagonal_probv,
)
from mqns.models.core.operator import OPERATOR_PAULI_I, OPERATOR_PAULI_X, OPERATOR_PAULI_Y, OPERATOR_PAULI_Z, Operator
from mqns.models.core.state import ATOL
from mqns.models.error.error import ErrorModel

OP_IZXY = [OPERATOR_PAULI_I, OPERATOR_PAULI_Z, OPERATOR_PAULI_X, OPERATOR_PAULI_Y]


class PauliErrorModelBase(ErrorModel):
    _probv0 = make_bell_diagonal_probv(1, 0, 0, 0)

    def __init__(self, name: str):
        super().__init__(name)

        self.probv = self._probv0
        """Probability of I,Z,X,Y result."""
        # initial p_survival is 1.0, corresponding to probv=[1,0,0,0]

    @abstractmethod
    def _prepare(self) -> None:
        """
        Subclass must invoke _set_probv() in this method.
        """

    def _set_probv(self, probv: BellDiagonalProbV) -> None:
        self.probv = probv

        try:
            del self._ptm
        except AttributeError:
            pass

    @functools.cached_property
    def _ptm(self) -> PauliTransferMat:
        """
        Construct the transition matrix for Bell-diagonal states.
        """
        return bell_diagonal_probv_to_pauli_transfer_mat(self.probv)

    @override
    def werner(self, q) -> None:
        q.w *= self.p_survival

    @override
    def mixed(self, q) -> None:
        q.set_probv(self._ptm @ q.probv, copy=False)


class PauliErrorModel(PauliErrorModelBase):
    """
    Pauli error model that maps channel degradation into a combination of stochastic Pauli gates.

    This model interprets ``p_survival`` and ``p_error`` to scale individual gate application
    probabilities (Z, X, Y) toward an asymptotic mixed-state baseline as noise accumulates.
    """

    def __init__(self, name="pauli", *, z=0.0, x=0.0, y=0.0):
        """
        Constructor.

        Args:
            name: name of this error model.
            z: relative weight of Z-gate noise component.
            x: relative weight of X-gate noise component.
            y: relative weight of Y-gate noise component.
        """
        super().__init__(name)

        ratios: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]] = np.array([z, x, y], dtype=np.float64)
        total = np.sum(ratios)
        if total > ATOL:
            ratios /= total
        else:
            ratios.fill(1 / 3)

        # Determine the mixed-state baseline reachable if p_error is 1.
        active_axes = np.count_nonzero(ratios)
        p_i_floor = 1.0 / (1.0 + active_axes) if active_axes > 0 else 1.0
        error_floor = ratios * (1.0 - p_i_floor)
        self._asymptotic_floor = make_bell_diagonal_probv(p_i_floor, *error_floor)

    @override
    def _prepare(self) -> None:
        probv = self.p_error * self._asymptotic_floor
        probv[0] += self.p_survival
        self._set_probv(probv)

        try:
            del self._stochastic_ops
        except AttributeError:
            pass

    @functools.cached_property
    def _stochastic_ops(self) -> tuple[list[Operator], list[float]]:
        """
        Construct the operators and probabilities for single qubit stochastic operations.
        """
        ops: list[Operator] = []
        prob: list[float] = []
        for o, p in zip(OP_IZXY, cast(Iterable[float], self.probv), strict=True):
            if p > 0:
                ops.append(o)
                prob.append(p)
        assert len(ops) > 0
        return ops, prob

    @override
    def qubit(self, q) -> None:
        q.stochastic_operate(*self._stochastic_ops)


class DepolarErrorModel(PauliErrorModel):
    """
    Depolarizing error model representing isotropic channel degradation.

    As ``p_error`` approaches 1, the gate probabilities scale uniformly to drive
    single-qubit states toward the maximally mixed state, distributed evenly
    across Z, X, and Y operations.
    """

    def __init__(self, name="depolarizing"):
        super().__init__(name, z=1, x=1, y=1)


class DephaseErrorModel(PauliErrorModel):
    """
    Dephasing error model representing pure phase-damping noise.

    Isolates degradation along the longitudinal axis. As ``p_error`` approaches 1,
    the probabilities of the identity (I) and phase-flip (Z) operations scale to
    converge at an equal 50/50 split, completely destroying off-diagonal coherence
    without causing state oscillations.
    """

    def __init__(self, name="dephasing"):
        super().__init__(name, z=1)


class BitFlipErrorModel(PauliErrorModel):
    """
    Bit-flip error model representing amplitude inversion noise.

    Isolates degradation along the transversal axis. As ``p_error`` approaches 1,
    the probabilities of the identity (I) and bit-flip (X) operations scale to
    converge at an equal 50/50 split, completely mixing populations without causing
    deterministic state inversions.
    """

    def __init__(self, name="bit-flip"):
        super().__init__(name, x=1)
