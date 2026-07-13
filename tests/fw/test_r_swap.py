"""
Test suite for ReactiveForwarder focused on swapping.
"""

import itertools
from collections import defaultdict

import pytest

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.network.fw import RoutingController, RoutingPathStatic
from mqns.network.network import Request, TimingModeSync, TimingPhase, TimingPhaseEvent
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.reactive import ReactiveForwarder, ReactiveRoutingController
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.simulator import event_handler, func_to_event

from .fw_common import (
    build_linear_network,
    build_tree_network,
    check_fw_counters,
    print_node_counters,
    provide_entanglements,
)


class ManualController(ClassicCommandDispatcherMixin, RoutingController):
    def __init__(self):
        super().__init__(mv_auto="max")

        self.ls_pkts: list[tuple[ClassicPacket, LinkStateMsg]] = []
        self.ls_entries: list[LinkStateEntry] = []

    @event_handler
    def handle_sync_phase(self, event: TimingPhaseEvent):
        match event.action:
            case TimingPhase.ROUTING, False:
                self.ls_pkts.clear()
                self.ls_entries.clear()

    @classic_cmd_handler("LS")
    def handle_ls(self, pkt: ClassicPacket, msg: LinkStateMsg):
        _ = pkt
        self.ls_pkts.append((pkt, msg))
        self.ls_entries.extend(msg["ls"])


def test_tree2_one():
    """Verify link state messages and test one path in tree (height=2) topology."""
    ctrl = ManualController()
    net, simulator = build_tree_network(
        2,
        mode="R",
        ch_capacity=2,
        ctrl=ctrl,
        fw={"p_swap": 1.0},
        end_time=0.020,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
    )
    fwA, fwB, fwC, fwD, _, fwF, _ = (node.get_app(ReactiveForwarder) for node in net.nodes)

    def do_routing():
        assert len(ctrl.ls_pkts) == 5
        assert len(ctrl.ls_entries) == 8
        ctrl.install_path(RoutingPathStatic("DBACF", req_id=1, swap=[2, 0, 1, 0, 2]))

    for slot in 0.000, 0.010:
        t0 = slot
        simulator.add_event(func_to_event(simulator.time(sec=t0 + 0.0065), do_routing))
        provide_entanglements(
            (0.0011, fwD, fwB),
            (0.0012, fwB, fwA),
            (0.0013, fwA, fwC),
            (0.0014, fwC, fwF),
            transform_t=lambda s: t0 + s,
        )
    simulator.run()
    print_node_counters(net)

    reqDF = RequestCounters.of(net, 1, "D-F")
    assert reqDF.n_consumed == 2


def test_tree2_two():
    """Verify link state messages and test both paths in tree (height=2) topology."""
    ctrl = ManualController()
    net, simulator = build_tree_network(
        2,
        mode="R",
        ch_capacity=4,
        ctrl=ctrl,
        fw={"p_swap": 1.0},
        end_time=0.020,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
    )
    fwA, fwB, fwC, fwD, fwE, fwF, fwG = (node.get_app(ReactiveForwarder) for node in net.nodes)

    def do_routing():
        assert len(ctrl.ls_pkts) == 7
        assert len(ctrl.ls_entries) == 16

        qubits_by_channel = defaultdict[str, list[str]](lambda: [])
        for entry in ctrl.ls_entries:
            qubits_by_channel[f"{entry['node']}{entry['neighbor']}"].append(entry["qubit"])

        for i, route in enumerate(("DBACF", "EBACG")):
            ctrl.install_path(
                RoutingPathStatic(
                    route,
                    req_id=1 + i,
                    swap=[2, 0, 1, 0, 2],
                    m_v=[qubits_by_channel[f"{a}{b}"].pop() for a, b in itertools.pairwise(route)],
                )
            )

    for slot in 0.000, 0.010:
        t0 = slot
        simulator.add_event(func_to_event(simulator.time(sec=t0 + 0.0065), do_routing))
        provide_entanglements(
            (0.0011, fwD, fwB),
            (0.0012, fwB, fwA),
            (0.0013, fwA, fwC),
            (0.0014, fwC, fwF),
            (0.0021, fwE, fwB),
            (0.0022, fwB, fwA),
            (0.0023, fwA, fwC),
            (0.0024, fwC, fwG),
            transform_t=lambda s: t0 + s,
        )

    simulator.run()
    print_node_counters(net)

    reqDF = RequestCounters.of(net, 1, "D-F")
    assert reqDF.n_consumed == 2
    reqEG = RequestCounters.of(net, 2, "E-G")
    assert reqEG.n_consumed == 2


@pytest.mark.parametrize(
    ("req_active", "etgAB", "etgBC", "cnt"),
    [
        # Request is active in both slots, EPRs arrive in first slot, request satisfied.
        ((0, 0.020), [0.001], [0.002], (3, 1)),
        # Request is active in both slots, EPRs arrive in second slot, request satisfied.
        ((0, 0.020), [0.011], [0.012], (3, 1)),
        # Request is active in both slots, EPRs arrive in both slots, request satisfied twice.
        ((0, 0.020), [0.001, 0.011], [0.002, 0.012], (6, 2)),
        # Request is active in both slots, EPRs arrive in separate slots, request unsatisfied.
        ((0, 0.020), [0.001], [0.012], (4, 0)),
        # Request is active in first slot, EPRs arrive in second slot, request unsatisfied.
        ((0, 0.010), [0.011], [0.012], (3, 0)),
        # Request is active in first slot, EPRs arrive twice in first slot, request satisfied twice.
        ((0, 0.010), [0.001, 0.003], [0.002, 0.004], (3, 2)),
    ],
)
def test_3_minimal(req_active: tuple[float, float], etgAB: list[float], etgBC: list[float], cnt: tuple[int, int]):
    """Test 3-node minimal swap, two time slots."""
    net, simulator = build_linear_network(
        3,
        ch_capacity=2,
        mode="R",
        fw={"p_swap": 1.0},
        end_time=0.020,
        timing=TimingModeSync(t_ext=0.006, t_rtg=0.001, t_int=0.003),
    )
    ctrl = net.get_controller().get_app(ReactiveRoutingController)
    fwA, fwB, fwC = (node.get_app(ReactiveForwarder) for node in net.nodes)

    net.add_request(Request("A-C", active_period=req_active).path(req_id=1))
    provide_entanglements(
        *((t, fwA, fwB) for t in etgAB),
        *((t, fwB, fwC) for t in etgBC),
    )
    simulator.run()
    print(ctrl.cnt)
    print_node_counters(net)

    assert (ctrl.cnt.n_ls, ctrl.cnt.n_satisfy) == cnt
    check_fw_counters(
        net,
        n_swapped=(0, cnt[1], 0),
    )
    assert RequestCounters.of(net, 1, "A-C").n_consumed == cnt[1]
