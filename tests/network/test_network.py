import itertools

import numpy as np
import pytest

from mqns.entity.node import Application, Controller, Node, NodePair, split_node_pair
from mqns.models.core import TrafficMatrixInput
from mqns.network.network import (
    MatrixTrafficGenerator,
    QuantumNetwork,
    RequestActiveEvent,
    RequestInactiveEvent,
    TimingModeSync,
    TimingPhase,
    TrafficMatrixMapping,
    sync_phase_handler,
)
from mqns.network.topology import BasicTopology
from mqns.simulator import Simulator, event_handler, func_to_event


class SyncCheckApp(Application[Node]):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        assert self.simulator.tc.time_slot % 5000 == 0
        self.records.append("E+")

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        assert self.simulator.tc.time_slot % 5000 == 4000
        self.records.append("E-")

    @sync_phase_handler(TimingPhase.INTERNAL, True)
    def sync_internal_enter(self) -> None:
        assert self.simulator.tc.time_slot % 5000 == 4000
        self.records.append("I+")

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        assert self.simulator.tc.time_slot % 5000 == 0
        self.records.append("I-")


def test_timing_mode_sync():
    topo = BasicTopology(2, nodes_apps=[SyncCheckApp()])
    topo.controller = Controller("ctrl", apps=[SyncCheckApp()])
    timing = TimingModeSync(t_ext=4, t_int=1)
    net = QuantumNetwork(topo, timing=timing)

    s = Simulator(0.0, 29.9, accuracy=1000, install_to=(net,))
    assert set(node.name for node in net.all_nodes) == {"ctrl", "n1", "n2"}
    assert timing.t_ext.time_slot == 4000
    assert timing.t_rtg.time_slot == 0
    assert timing.t_int.time_slot == 1000

    s.run()

    for node in net.all_nodes:
        app = node.get_app(SyncCheckApp)
        assert app.records == ["E+", "E-", "I+", "I-"] * 5 + ["E+", "E-", "I+"]


@pytest.mark.parametrize(
    ("tm", "src_dst"),
    [
        pytest.param([[0, 1, 0], [0, 0, 0], [0, 0, 0]], "A-B", id="list[list]"),
        pytest.param(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), "B-C", id="ndarray"),
        pytest.param({"C-A": 1}, "C-A", id="dict"),
    ],
)
def test_mtg_eager(tm: TrafficMatrixInput | TrafficMatrixMapping, src_dst: NodePair):
    topo = BasicTopology(3, nodes_naming="A")
    net = QuantumNetwork(topo)

    mtg = MatrixTrafficGenerator(net, tm, rate=10, sched="eager")
    Simulator(100, 200, install_to=(net, mtg))

    # expect 1000 requests in 100 seconds for 10Hz
    assert 800 <= len(net.requests) <= 1200

    src_dst = split_node_pair(src_dst)
    for req in net.requests:
        assert (req.src, req.dst) == src_dst

    for req0, req1 in itertools.pairwise(net.requests):
        assert req0.active_since <= req1.active_until


class RequestCheckApp(Application[Controller]):
    net: QuantumNetwork

    def __init__(self):
        super().__init__()
        self.enters: list[int] = []
        """Length of ``net.requests`` upon entering each request."""
        self.exits = 0

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req
        assert (req.src, req.dst) == ("A", "C")
        assert req.is_active(event.t) is True
        self.enters.append(len(self.net.requests))

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        req = event.req
        assert (req.src, req.dst) == ("A", "C")
        assert req.is_active(event.t) is False
        self.exits += 1


def test_mtg_lazy():
    topo = BasicTopology(3, nodes_naming="A")
    topo.controller = Controller("ctrl", apps=[app := RequestCheckApp()])
    net = QuantumNetwork(topo)
    app.net = net

    mtg = MatrixTrafficGenerator(net, {"A-C": 1}, rate=10, sched="lazy")
    s = Simulator(100, np.inf, need_synchronized=False, install_to=(net, mtg))
    s.add_event(func_to_event(s.time(sec=200), s.stop))
    s.run()

    # expect 1000 requests in 100 seconds for 10Hz
    assert 800 <= len(app.enters) <= 1200

    # no more than 100 extra requests upon entering each request
    for i, n_requests in enumerate(app.enters):
        assert n_requests <= i + 100
