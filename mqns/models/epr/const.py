import numpy as np

from mqns.models.qubit import state_to_rho
from mqns.models.qubit.typing import MultiQubitState

QS_PHI_P: MultiQubitState = np.array([[1], [0], [0], [1]], dtype=np.complex128) / np.sqrt(2)
"""Bell basis ``Phi^+`` state."""
QS_PHI_N: MultiQubitState = np.array([[1], [0], [0], [-1]], dtype=np.complex128) / np.sqrt(2)
"""Bell basis ``Phi^-`` state."""
QS_PSI_P: MultiQubitState = np.array([[0], [1], [1], [0]], dtype=np.complex128) / np.sqrt(2)
"""Bell basis ``Psi^+`` state."""
QS_PSI_N: MultiQubitState = np.array([[0], [1], [-1], [0]], dtype=np.complex128) / np.sqrt(2)
"""Bell basis ``Psi^-`` state."""

RHO_PHI_P = state_to_rho(QS_PHI_P)
"""Bell basis ``Phi^+`` density matrix."""
RHO_PHI_N = state_to_rho(QS_PHI_N)
"""Bell basis ``Phi^-`` density matrix."""
RHO_PSI_P = state_to_rho(QS_PSI_P)
"""Bell basis ``Psi^+`` density matrix."""
RHO_PSI_N = state_to_rho(QS_PSI_N)
"""Bell basis ``Psi^-`` density matrix."""
