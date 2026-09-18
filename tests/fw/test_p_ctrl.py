"""
Test suite for proactive forwarding focused on control plane.
"""

import pytest

from mqns.network.fw import RoutingPath
from mqns.network.network import Request, RequestState

from .fw_common import build_grid_network


def test_rect2_ru():
    """Test resource utilization tracking in 2x2 rectangular topology."""
    net, simulator = build_grid_network(t_cohere=0.005, ch_capacity=3, drq_cap=0)
    # FIB erase delay is 0.020 seconds.

    # t=0.000, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req0 := Request(RoutingPath.static("ABD", "ACD", bufferspace_mv=1), active_period=(1.001, 2.001)))
    # t=1.001, available qubits per link: AB=2, AC=2, BD=2, CD=2.
    net.add_request(req1 := Request(RoutingPath.static("ABD", bufferspace_mv=1), active_period=(1.002, 2.002)))
    # t=1.002, available qubits per link: AB=0, AC=2, BD=0, CD=2.
    net.add_request(req2 := Request(RoutingPath.static("ABD", bufferspace_mv=2), active_period=(1.003, 2.003)))
    # req2 is rejected due to insufficient resources on AB and BD links.
    net.add_request(req3 := Request(RoutingPath.static("ACD", bufferspace_mv=2), active_period=(1.004, 2.004)))
    # t=1.004, available qubits per link: AB=0, AC=0, BD=0, CD=0.

    # req0 releases its qubits at 2.021; req3 releases its qubits at 2.024.
    # t=2.023, req0,req1,req2 released, available qubits per link: AB=3, AC=1, BD=3, CD=1.
    net.add_request(req4 := Request(RoutingPath.static("BAC", bufferspace_mv=3), active_period=(2.023, 3.023)))
    # req4 is rejected due to insufficient resources on AC link.
    # t=2.024, req3 released, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req5 := Request(RoutingPath.static("AC", bufferspace_mv=3), active_period=(2.025, 3.025)))

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert req1.state is RequestState.EXPIRED
    assert req2.state is RequestState.REJECTED
    assert req3.state is RequestState.EXPIRED
    assert req4.state is RequestState.REJECTED
    assert req5.state is RequestState.EXPIRED


@pytest.mark.parametrize(
    ("mv2", "state3"),
    [
        # t=1.620, req1 released, available qubits per link: AB=1, AC=2, BD=1, CD=2.
        # req3 is accepted at this time.
        (2, RequestState.EXPIRED),
        # t=1.620, req1 released, available qubits per link: AB=0, AC=2, BD=0, CD=2.
        # There's still no resources to accept req3.
        # t=2.004, req3 is rejected when it reaches expiration.
        (3, RequestState.REJECTED),
    ],
)
def test_rect2_drq(mv2: int, state3: RequestState):
    """Test deferred request queue in 2x2 rectangular topology."""
    net, simulator = build_grid_network(t_cohere=0.005, ch_capacity=3, drq_cap=2)
    # FIB erase delay is 0.020 seconds.

    # t=0.000, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req0 := Request(RoutingPath.static("ABD", bufferspace_mv=3), active_period=(1.001, 1.500)))
    net.add_request(req1 := Request(RoutingPath.static("ACD", bufferspace_mv=2), active_period=(1.002, 1.600)))
    # t=1.002, available qubits per link: AB=0, AC=1, BD=0, CD=1.
    net.add_request(req2 := Request(RoutingPath.static("ABD", bufferspace_mv=mv2), active_period=(1.003, 2.003)))
    # req2 is deferred due to insufficient resources on AB and BD links.
    net.add_request(req3 := Request(RoutingPath.static("ABD", "ACD", bufferspace_mv=1), active_period=(1.004, 2.004)))
    # req3 is deferred due to insufficient resources on AB and BD links.
    net.add_request(req4 := Request(RoutingPath.static("ABD", bufferspace_mv=1), active_period=(1.005, 2.005)))
    # req4 is rejected because the queue is full.
    net.add_request(req5 := Request(RoutingPath.static("ACD", bufferspace_mv=1), active_period=(1.006, 2.006)))
    # t=1.006, available qubits per link: AB=0, AC=0, BD=0, CD=0.

    # t=1.520, req0 released, available qubits per link: AB=3, AC=0, BD=3, CD=0.
    # req2 is accepted at this time.
    # See per-parameter-set comments for fate of req3.

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert req1.state is RequestState.EXPIRED
    assert req2.state is RequestState.EXPIRED
    assert req3.state is state3
    assert req4.state is RequestState.REJECTED
    assert req5.state is RequestState.EXPIRED
