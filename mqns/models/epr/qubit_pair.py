from typing import Unpack, override

import numpy as np

from mqns.models.core.state import qubit_rho_remove
from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs
from mqns.models.error import PauliErrorModel
from mqns.models.qubit import Qubit
from mqns.models.qubit.gate import CNOT, H


class EntangledQubitPair(Entanglement):
    """
    Entanglement model represented with two ``Qubit`` instances.

    This model is primarily intended for numeric verification of other models.
    """

    q0: Qubit
    q1: Qubit

    def __init__(self):
        self.q0 = Qubit()
        self.q1 = Qubit()
        H(self.q0)
        CNOT(self.q0, self.q1)

    @property
    @override
    def fidelity(self) -> float:
        if self.q0.state is not self.q1.state:
            assert self.is_decohered
            return 0.0

        state = self.q0.state
        rho = state.rho
        idx0 = state.qubits.index(self.q0)
        idx1 = state.qubits.index(self.q1)
        if (n := state.num) > 2:
            # Trace out other qubits (if any) from the density matrix.
            rho = qubit_rho_remove(rho, (i for i in range(n) if i not in (idx0, idx1)), n)
        if idx1 < idx0:
            # If the density matrix is in [q1,q0] order, swap the qubits.
            rho = rho.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

        # Compare to the desired state BELL_RHO_PHI_P.
        f_val = 0.5 * (rho[0, 0] + rho[0, 3] + rho[3, 0] + rho[3, 3])
        return np.real(f_val)

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        _ = value
        raise TypeError("cannot assign fidelity to EntangledQubitPair")

    @staticmethod
    @override
    def _make_swapped(epr0: "EntangledQubitPair", epr1: "EntangledQubitPair", **kwargs: Unpack[EntanglementInitKwargs]):
        raise NotImplementedError()

    @override
    def _do_purify(self, epr1: "EntangledQubitPair") -> bool:
        raise NotImplementedError()

    @override
    def apply_error(self, error) -> None:
        if isinstance(error, PauliErrorModel):
            self.q0.apply_error(error)
        else:
            raise TypeError("non-Pauli error must be applied to an individual qubit")
