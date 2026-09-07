"""
Test suite for proactive forwarding focused on control plane.
"""

from mqns.network.fw import RoutingPathStatic
from mqns.network.network import Request, RequestState

from .fw_common import build_grid_network


def test_rect2_ru():
    """Test resource utilization tracking in 2x2 rectangular topology."""
    net, simulator = build_grid_network(k_paths=2, t_cohere=0.005, ch_capacity=3)
    # FIB erase delay is 0.020 seconds.

    # t=0.000, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req0 := Request(RoutingPathStatic("ABD", "ACD", bufferspace_mv=1), active_period=(1.001, 2.001)))
    # t=1.001, available qubits per link: AB=2, AC=2, BD=2, CD=2.
    net.add_request(req1 := Request(RoutingPathStatic("ABD", bufferspace_mv=1), active_period=(1.002, 2.002)))
    # t=1.002, available qubits per link: AB=0, AC=2, BD=0, CD=2.
    net.add_request(req2 := Request(RoutingPathStatic("ABD", bufferspace_mv=2), active_period=(1.003, 2.003)))
    # req2 is rejected due to insufficient resources on AB and BD links.
    net.add_request(req3 := Request(RoutingPathStatic("ACD", bufferspace_mv=2), active_period=(1.004, 2.004)))
    # t=1.004, available qubits per link: AB=0, AC=0, BD=0, CD=0.

    # req0 releases its qubits at 2.021; req3 releases its qubits at 2.024.
    # t=2.023, req0,req1,req2 released, available qubits per link: AB=3, AC=1, BD=3, CD=1.
    net.add_request(req4 := Request(RoutingPathStatic("BAC", bufferspace_mv=3), active_period=(2.023, 3.023)))
    # req4 is rejected due to insufficient resources on AC link.
    # t=2.024, req3 released, available qubits per link: AB=3, AC=3, BD=3, CD=3.
    net.add_request(req5 := Request(RoutingPathStatic("AC", bufferspace_mv=3), active_period=(2.025, 3.025)))

    simulator.run()

    assert req0.state is RequestState.EXPIRED
    assert req1.state is RequestState.EXPIRED
    assert req2.state is RequestState.REJECTED
    assert req3.state is RequestState.EXPIRED
    assert req4.state is RequestState.REJECTED
    assert req5.state is RequestState.EXPIRED
