#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Iterable
from typing import Unpack, final, overload, override

import numpy as np

from mqns.models.epr.entanglement import Entanglement, EntanglementInitKwargs
from mqns.models.qubit.state import BELL_RHO_PHI_N, BELL_RHO_PHI_P, BELL_RHO_PSI_N, BELL_RHO_PSI_P, QubitRho, check_qubit_rho
from mqns.utils import get_rand


@final
class MixedStateEntanglement(Entanglement["MixedStateEntanglement"]):
    """A pair of entangled qubits in Bell-Diagonal State with a hidden-variable."""

    @overload
    def __init__(self, *, fidelity=1.0, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with fidelity.

        This creates a Werner state where the error probabilities are distributed equals among the three bad states.
        """
        pass

    @overload
    def __init__(self, *, i: float, z: float, x: float, y: float, **kwargs: Unpack[EntanglementInitKwargs]):
        """
        Construct with four probability values.

        Args:
            i: Probability of desired state, i.e. fidelity.
            z: Probability of Z-flip.
            x: Probability of X-flip.
            y: Probability of Y-flip.
        """
        pass

    def __init__(self, *, fidelity: float | None = None, i=1.0, z=0.0, x=0.0, y=0.0, **kwargs: Unpack[EntanglementInitKwargs]):
        super().__init__(**kwargs)
        if fidelity is None:
            self._set_probabilities(i, z, x, y)
        else:
            bcd = (1 - fidelity) / 3
            self._set_probabilities(fidelity, bcd, bcd, bcd)

    @property
    @override
    def fidelity(self) -> float:
        return self.i

    @fidelity.setter
    @override
    def fidelity(self, value: float):
        assert 0 <= value <= 1
        self.i = value

    def _set_probabilities(self, i: float, z: float, x: float, y: float) -> None:
        total = i + z + x + y
        if total <= 0:  # avoid divide-by-zero
            inv_total = 0
        else:
            inv_total = 1 / total
        self.i = i * inv_total
        """Probability of desired state."""
        self.z = z * inv_total
        """Probability of Z-flip."""
        self.x = x * inv_total
        """Probability of X-flip."""
        self.y = y * inv_total
        """Probability of Y-flip."""

    @staticmethod
    @override
    def _make_swapped(epr0: "MixedStateEntanglement", epr1: "MixedStateEntanglement", **kwargs: Unpack[EntanglementInitKwargs]):
        return MixedStateEntanglement(
            i=epr0.i * epr1.i + epr0.z * epr1.z + epr0.x * epr1.x + epr0.y * epr1.y,
            z=epr0.i * epr1.z + epr0.z * epr1.i + epr0.x * epr1.y + epr0.y * epr1.x,
            x=epr0.i * epr1.x + epr0.z * epr1.y + epr0.x * epr1.i + epr0.y * epr1.z,
            y=epr0.i * epr1.y + epr0.z * epr1.x + epr0.x * epr1.z + epr0.y * epr1.i,
            **kwargs,
        )

    @override
    def _do_purify(self, epr1: "MixedStateEntanglement") -> bool:
        """
        Perform distillation using BBPSSW protocol.
        """
        p_succ = (self.i + self.y) * (epr1.i + epr1.y) + (self.z + self.x) * (epr1.x + epr1.z)
        if get_rand() > p_succ:
            return False

        self._set_probabilities(
            self.i * epr1.i + self.y * epr1.y,
            self.z * epr1.z + self.x * epr1.x,
            self.z * epr1.x + self.x * epr1.z,
            self.i * epr1.y + self.y * epr1.i,
        )
        return True

    def dephase(self, t: float, rate: float):
        """
        Inject dephasing noise.

        Args:
            t: time in seconds, distance in km, etc.
            rate: dephasing rate, unit is inverse of ``t``.
        """
        multiplier = np.exp(-rate * t)
        self._set_probabilities(
            0.5 + (self.i - 0.5) * multiplier,
            0.5 + (self.z - 0.5) * multiplier,
            self.x,
            self.y,
        )

    def depolarize(self, t: float, rate: float):
        """
        Inject depolarizing noise.

        Args:
            t: time in seconds, distance in km, etc.
            rate: depolarizing rate, unit is inverse of ``t``.
        """
        multiplier = np.exp(-rate * t)
        self._set_probabilities(
            0.25 + (self.i - 0.25) * multiplier,
            0.25 + (self.z - 0.25) * multiplier,
            0.25 + (self.x - 0.25) * multiplier,
            0.25 + (self.y - 0.25) * multiplier,
        )

    @override
    def store_error_model(self, t: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        The default error model for storing this entangled pair in a quantum memory.
        The default behavior is::

            a = 0.25 + (a-0.25)*e^{decoherence_rate*t}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*t}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*t}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*t}

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate: the decoherence rate, equals to 1/T_coh, where T_coh is the coherence time.
            kwargs: other parameters

        """
        _ = kwargs
        self.depolarize(t, decoherence_rate)

    @override
    def transfer_error_model(self, length: float = 0, decoherence_rate: float = 0, **kwargs):
        """
        The default error model for transmitting this entanglement.
        The success possibility of transmitting is::

            a = 0.25 + (a-0.25)*e^{decoherence_rate*length}
            b = 0.25 + (b-0.25)*e^{decoherence_rate*length}
            c = 0.25 + (c-0.25)*e^{decoherence_rate*length}
            d = 0.25 + (d-0.25)*e^{decoherence_rate*length}

        Args:
            length (float): the length of the channel
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters

        """
        _ = kwargs
        self.depolarize(length, decoherence_rate)

    @override
    def _to_qubits_rho(self) -> QubitRho:
        return check_qubit_rho(
            self.i * BELL_RHO_PHI_P + self.z * BELL_RHO_PHI_N + self.x * BELL_RHO_PSI_P + self.y * BELL_RHO_PSI_N, n=2
        )

    @override
    def _describe_fidelity(self) -> Iterable[str]:
        yield f"i={self.i:.4f}"
        yield f"z={self.z:.4f}"
        yield f"x={self.x:.4f}"
        yield f"y={self.y:.4f}"
