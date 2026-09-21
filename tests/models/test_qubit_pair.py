import pytest

from mqns.models.epr import EntangledQubitPair, Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error import DepolarErrorModel
from mqns.models.qubit import QState


@pytest.mark.parametrize("epr_type", [EntangledQubitPair, MixedStateEntanglement, WernerStateEntanglement])
def test_fidelity_error(epr_type: type[Entanglement]):
    epr = epr_type()
    assert epr.fidelity == pytest.approx(1.0, abs=1e-6)

    depolar = DepolarErrorModel().set(p_error=0.1)
    epr.apply_error(depolar)
    assert epr.fidelity == pytest.approx(0.925, abs=1e-6)


def test_fidelity_trace():
    epr0 = EntangledQubitPair()
    epr0.apply_error(DepolarErrorModel().set(p_error=0.1))

    epr1 = EntangledQubitPair()
    QState.joint(epr0.q0, epr1.q0)
    epr1.apply_error(DepolarErrorModel().set(p_error=0.2))
    assert epr0.q0.state.num == 4

    assert epr0.fidelity == pytest.approx(0.925, abs=1e-6)
    assert epr1.fidelity == pytest.approx(0.850, abs=1e-6)
    assert epr0.q0.state.num == 4
