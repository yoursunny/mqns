"""
Test suite for swapping in proactive forwarding, standalone without LinkLayer.
"""

import pytest

from mqns.network.proactive import (
    ProactiveForwarder,
    ProactiveRoutingController,
    RoutingPathSingle,
)

from .proactive_common import (
    build_linear_network,
    install_path,
    provide_entanglements,
)


@pytest.mark.parametrize(
    ("arrival_ms", "n_swapped_p"),
    [
        ((1, 2, 1), 1),
        ((2, 1, 2), 1),
        ((1, 2, 3), 0),
        ((3, 2, 1), 0),
    ],
)
def test_swap_asap(arrival_ms: tuple[int, int, int], n_swapped_p: int):
    """Test SWAP-ASAP with various entanglement arrival orders."""
    net, simulator = build_linear_network(4, ps=1.0, has_link_layer=False)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathSingle("n1", "n4", swap=[1, 0, 0, 1]))
    provide_entanglements(
        (1 + arrival_ms[0] / 1000, f1, f2),
        (1 + arrival_ms[1] / 1000, f2, f3),
        (1 + arrival_ms[2] / 1000, f3, f4),
    )
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_consumed == 1 == f4.cnt.n_consumed
    assert f2.cnt.n_swapped_s == 1 == f3.cnt.n_swapped_s
    assert f2.cnt.n_swapped_p == n_swapped_p == f3.cnt.n_swapped_p
