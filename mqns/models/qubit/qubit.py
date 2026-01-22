from typing import overload

from mqns.models.core import BASIS_Z, MeasureOutcome, Operator, QuantumModel, QubitRho, QubitState
from mqns.models.core.state import QUBIT_RHO_0, QUBIT_STATE_0
from mqns.models.qubit.state import QState


class Qubit(QuantumModel):
    """Represent a qubit."""

    @overload
    def __init__(
        self,
        state: QubitState = QUBIT_STATE_0,
        *,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        """
        Construct with qubit state.

        Args:
            state: initial state, default is ``|0>``.
            operate_decoherence_rate: operate decoherence rate.
            measure_decoherence_rate: measure decoherence rate.
            name: descriptive name.
        """

    @overload
    def __init__(
        self,
        *,
        rho: QubitRho,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        """
        Construct with density matrix.

        Args:
            state: initial density matrix.
            operate_decoherence_rate: operate decoherence rate.
            measure_decoherence_rate: measure decoherence rate.
            name: descriptive name.
        """

    def __init__(
        self,
        state: QubitState | None = None,
        *,
        rho: QubitRho = QUBIT_RHO_0,
        operate_decoherence_rate=0.0,
        measure_decoherence_rate=0.0,
        name="",
    ):
        self.name = name
        """Descriptive name."""
        self.state = QState([self], state=state, rho=rho)
        """QState that includes this qubit."""
        self.operate_decoherence_rate = operate_decoherence_rate
        """Operate decoherence rate."""
        self.measure_decoherence_rate = measure_decoherence_rate
        """Measure decoherence rate."""

    def measure(self, basis=BASIS_Z) -> MeasureOutcome:
        """
        Measure this qubit with the specified basis.

        Args:
            basis: Measurement basis, defaults to Z.

        Returns: Measurement outcome 0 or 1.
        """
        self.measure_error_model(decoherence_rate=self.measure_decoherence_rate)
        return self.state.measure(self, basis)

    def stochastic_operate(self, operators: list[Operator] = [], probabilities: list[float] = []) -> None:
        """
        Apply a set of operators with associated probabilities to the qubit.
        It usually turns a pure state into a mixed state.

        Args:
            operators: a list of operators, each must operate on a single qubit.
            probabilities: the probability of applying each operator; their sum must be 1.
        """
        i, n = self.state.qubits.index(self), self.state.num
        full_operators: list[Operator] = [op.lift(i, n) for op in operators]
        self.state.stochastic_operate(full_operators, probabilities)

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit " + self.name + ">"
        return super().__repr__()
