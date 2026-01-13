import numpy as np

QubitState = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Single qubit state, shape is (2,1)."""

MultiQubitState = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Multiple qubit state, shape is (2*n,1)."""

QubitRho = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Single qubit density matrix, shape is (2,2)."""

MultiQubitRho = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Multiple qubit density matrix, shape is (2*n,2*n)."""

Operator = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Operator on multiple qubits, shape is (2*n,2*n)."""

Operator1 = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Operator on single qubit, shape is (2,2)."""

Operator2 = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Operator on two qubits, shape is (4,4)."""

Basis = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
"""Measurement basis, shape is (2,2)."""
