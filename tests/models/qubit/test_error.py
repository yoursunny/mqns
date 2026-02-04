import numpy as np
import pytest

from mqns.models.core.state import (
    QUBIT_RHO_0,
    QUBIT_STATE_0,
    QUBIT_STATE_1,
    QUBIT_STATE_P,
    QubitState,
    check_qubit_rho,
    qubit_rho_equal,
    qubit_state_equal,
)
from mqns.models.error import CoherentErrorModel, DissipationErrorModel
from mqns.models.error.input import ErrorModelInputBasic
from mqns.models.qubit import Qubit
from mqns.models.qubit.gate import CNOT, H
from mqns.utils import rng


@pytest.mark.parametrize(
    ("error", "success", "success_atol"),
    [
        (None, 1000, 0),
        ({"p_error": 0.06}, 960, 30),  # should have 96% success rate
    ],
)
def test_measure_error(error: ErrorModelInputBasic, success: int, success_atol: int):
    cnt = 0
    for _ in range(1000):
        q0 = Qubit(measure_error=error)
        if q0.measure() == 0:
            cnt += 1
    assert np.isclose(cnt, success, atol=success_atol)


@pytest.mark.parametrize(
    ("error", "rho_input"),
    [
        (None, [[0.5, 0.5], [0.5, 0.5]]),
        ({"p_error": 0.18}, [[0.5, 0.38], [0.38, 0.5]]),
    ],
)
def test_operate_error(error: ErrorModelInputBasic, rho_input: np.typing.ArrayLike):
    q0 = Qubit(operate_error=error)
    H(q0)
    assert qubit_rho_equal(q0.state.rho, check_qubit_rho(np.array(rho_input, dtype=np.complex128)))


@pytest.mark.parametrize(
    ("operate_error", "measure_error", "success", "success_atol"),
    [
        (None, None, 100, 0),
        ({"p_error": 0.18}, {"p_error": 0.06}, 82, 15),  # should have 82.78% success rate
    ],
)
def test_entanglement(
    operate_error: ErrorModelInputBasic, measure_error: ErrorModelInputBasic, success: int, success_atol: int
):
    cnt = 0
    for _ in range(100):
        q0 = Qubit(operate_error=operate_error, measure_error=measure_error)
        q1 = Qubit(operate_error=operate_error, measure_error=measure_error)
        H(q0)
        CNOT(q0, q1)
        if q0.measure() == q1.measure():
            cnt += 1
    assert np.isclose(cnt, success, atol=success_atol)


@pytest.mark.parametrize(
    ("random", "decohered"),
    [
        (1.0, False),
        (0.0, True),
    ],
)
def test_dissipation(monkeypatch: pytest.MonkeyPatch, random: float, decohered: bool):
    q0 = Qubit(name="q0")
    q1 = Qubit(name="q1")
    H(q0)
    CNOT(q0, q1)

    error = DissipationErrorModel().set(p_error=0.1)
    monkeypatch.setattr(rng, "random", lambda: random)
    q0.apply_error(error)

    if decohered:
        assert q0.state is not q1.state
        assert qubit_rho_equal(q0.state.rho, QUBIT_RHO_0)
        assert q1.state.rho.diagonal() == pytest.approx([0.5, 0.5])
    else:
        assert q0.state is q1.state


@pytest.mark.parametrize(
    ("random", "state"),
    [
        (0.0, QUBIT_STATE_0),
        (0.5, QUBIT_STATE_P),
        (1.0, QUBIT_STATE_1),
    ],
)
def test_coherent(monkeypatch: pytest.MonkeyPatch, random: float, state: QubitState):
    q = Qubit()

    error = CoherentErrorModel(length=100, standard_lkm=50)
    monkeypatch.setattr(rng, "uniform", lambda a, b: a + random * (b - a))
    q.apply_error(error)

    pure_state = q.state.state()
    assert pure_state is not None
    assert qubit_state_equal(pure_state, state)
