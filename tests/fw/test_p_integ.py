"""
Test suite for ProactiveForwarder integrated with LinkLayer.
"""

import pytest

from mqns.entity.timer import Timer
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.fw import RoutingPathSingle, RoutingPathStatic, SwapSequenceInput
from mqns.network.network import TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import ProactiveForwarder
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.protocol.link_layer import LinkLayer

from .fw_common import build_linear_network, build_rect_network, install_path, print_node_counters


@pytest.mark.parametrize("epr_type", [WernerStateEntanglement, MixedStateEntanglement])
@pytest.mark.parametrize("timing", [TimingModeAsync(), TimingModeSync(t_ext=0.006, t_int=0.004)], ids=["ASYNC", "SYNC"])
@pytest.mark.parametrize("swap", ["asap", "l2r", "r2l"])
def test_4_swap(epr_type: type[Entanglement], timing: TimingMode, swap: SwapSequenceInput):
    """Test swapping in 4-node topology."""
    net, simulator = build_linear_network(
        4, swap_table_leak_tol=256, end_time=3.0, timing=timing, epr_type=epr_type, has_link_layer=True
    )
    _, fwB, fwC, _ = (node.get_app(ProactiveForwarder) for node in net.nodes)

    rp = install_path(net, RoutingPathSingle("A", "D", swap=swap))
    simulator.run()
    print_node_counters(net)

    # The main purpose of integrated test is to verify that the forwarder can return released qubits back to LinkLayer
    # for re-generating elementary entanglements.
    # Hence, these numeric bounds are much smaller than usual values, but must be greater than the memory capacity.
    assert fwB.cnt.n_swapped >= 16
    assert fwC.cnt.n_swapped >= 16
    assert RequestCounters.of(net, rp).n_consumed >= 16


def test_rect2_uninstall_path():
    """Test uninstall_path in 2x2 rectangle topology."""
    net, simulator = build_rect_network(t_cohere=1.0, has_link_layer=True)
    _, fwB, fwC, _ = (node.get_app(ProactiveForwarder) for node in net.nodes)
    llA, llB, llC, _ = (node.get_app(LinkLayer) for node in net.nodes)

    counters: list[tuple[int, int, int, int, int]] = []

    def save_counters():
        print_node_counters(net)
        counters.append(
            (
                fwB.cnt.n_swapped,
                llB.cnt.n_attempts,
                fwC.cnt.n_swapped,
                llC.cnt.n_attempts,
                llA.cnt.n_attempts,
            )
        )

    timer = Timer("save_counters", start_time=0.500, end_time=9.501, step_time=1.000, trigger_func=save_counters)
    timer.install(simulator)

    install_path(net, RoutingPathStatic("ABD"), t_install=2, t_uninstall=6)
    install_path(net, RoutingPathStatic("ACD"), t_install=4, t_uninstall=8)
    simulator.run()

    assert len(counters) == 10
    for i in 0, 1:  # fwB.cnt.n_swapped and llB.cnt.n_attempts
        assert counters[0][i] == counters[1][i]
        assert counters[6][i] == counters[9][i]
    for i in 2, 3:  # fwC.cnt.n_swapped and llC.cnt.n_attempts
        assert counters[0][i] == counters[3][i]
        assert counters[8][i] == counters[9][i]
    # llA.cnt.n_attempts
    assert counters[0][4] == counters[1][4]
    assert counters[8][4] == counters[9][4]
