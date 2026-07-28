from collections import deque
from typing import override

import pytest

from mqns.entity.memory import MemoryDecohereEvent, MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, LinkArchSr
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.network import QuantumNetwork, TimingModeSync
from mqns.network.protocol.event import ManageActiveChannel, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.protocol.link_layer import LinkLayer, LinkLayerCounters
from mqns.network.topology import ClassicTopology, CustomTopology, LinearTopology
from mqns.simulator import Simulator, event_handler, func_to_event
from mqns.utils import log


class NetworkLayer(Application[QNode]):
    def __init__(self):
        super().__init__()
        self.release_after = deque[float | None]()
        """If non-empty, ``QubitReleasedEvent`` would be emitted after specified duration for the next entanglement."""
        self.entangle: list[tuple[float, float]] = []
        """Entanglement events, each entry contains entanglement time and EPR creation time."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    @event_handler
    def handle_entangle(self, event: QubitEntangledEvent):
        mq, epr = self.memory.read(event.qubit.addr, has=self.epr_type)
        assert mq is event.qubit
        t_create = epr.decohere_time - self.memory.t_cohere
        self.entangle.append((event.t.sec, t_create.sec))

        try:
            release_after = self.release_after.popleft()
        except IndexError:
            return
        if release_after is not None:
            self.simulator.add_event(func_to_event(event.t + release_after, self._release, mq))

    def _release(self, mq: MemoryQubit):
        self.memory.read(mq.addr, remove=True)
        mq.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.node, mq, t=self.simulator.tc))

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent):
        self.decohere.append(event.t.sec)
        event.qubit.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.node, event.qubit, is_decoh=True, t=event.t))


def manage_active_channel(t: float, src: NetworkLayer, dst: NetworkLayer, *, start=True):
    simulator = src.simulator
    ch = src.node.get_qchannel(dst.node)
    time = simulator.time(sec=t)
    simulator.add_event(ManageActiveChannel(src.node, ch, path_id=None, start=start, is_primary=True, t=time))
    simulator.add_event(ManageActiveChannel(dst.node, ch, path_id=None, start=start, is_primary=False, t=time))


@pytest.mark.parametrize("epr_type", [WernerStateEntanglement, MixedStateEntanglement])
def test_basic(epr_type: type[Entanglement]):
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1, "link_arch": LinkArchAlways(LinkArchDimBk())},
        cchannel_args={"delay": 0.1},
        memory_args={"t_cohere": 4.1},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, epr_type=epr_type)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 20.0, install_to=(log, net))

    ll1, ll2 = (node.get_app(LinkLayer) for node in net.nodes)
    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)
    manage_active_channel(0.5, nl1, nl2)
    manage_active_channel(8.8, nl1, nl2, start=False)
    nl1.release_after.append(2.9)
    nl2.release_after.append(3.2)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)
        assert len(nl.entangle) == 3
        assert len(nl.decohere) == 2
        # t=0.5, path is installed with n1 as primary and n2 as secondary
        # t=0.5, n1 sends RESERVE_REQ
        # t=0.6, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=0.7, n1 receives RESERVE_RES
        # t=0.9, entanglement established
        # t=0.7 is assumed time of entanglement creation
        assert nl.entangle[0] == pytest.approx((0.9, 0.7), abs=1e-3)
        # t=3.8, n1 releases qubit and sends RESERVE_REQ
        # t=3.9, n2 receives RESERVE_REQ but has no qubit available
        # t=4.1, n2 releases qubit and sends RESERVE_RES
        # t=4.2, n1 receives RESERVE_RES
        # t=4.4, entanglement established
        # t=4.2 is assumed time of entanglement creation
        assert nl.entangle[1] == pytest.approx((4.4, 4.2), abs=1e-3)
        # t=8.3, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[0] == pytest.approx(8.3, abs=1e-3)
        # t=8.3, n1 sends RESERVE_REQ
        # t=8.4, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=8.5, n1 receives RESERVE_RES
        # t=8.7, entanglement established
        # t=8.5 is assumed time of entanglement creation
        assert nl.entangle[2] == pytest.approx((8.7, 8.5), abs=1e-3)
        # t=8.8, path is uninstalled
        # t=12.6, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[1] == pytest.approx(12.6, abs=1e-3)
        # no more entanglements because the path has been uninstalled

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert ll_cnt_agg.n_etg == 3
    assert ll_cnt_agg.n_attempts == 3
    # n_decoh is only incremented on the primary node.
    # Although there are two decoherence events in the simulation, l1.n_decoh is incremented only at t=8.3.
    # For the t=12.6 event, the path is uninstalled, so that l1 cannot recognize itself as primary.
    assert ll_cnt_agg.n_decoh == 1
    assert ll_cnt_agg.decoh_ratio == pytest.approx(1 / 3, abs=1e-6)

    QuantumMemory.check_leaks(net.nodes)


@pytest.mark.parametrize(
    ("uninstall_t", "qubits_state", "n_entangle"),
    [
        # don't uninstall, only check timeline states (expected qubits_state are ignored)
        (9.9, (QubitState.CONSUME, QubitState.CONSUME), (3, 3)),
        # uninstall when RESERVE_REQ is in-flight
        (0.2, (QubitState.RAW, QubitState.RAW), (0, 0)),
        # uninstall when RESERVE_RES is in-flight
        (0.5, (QubitState.RAW, QubitState.RAW), (0, 0)),
        # uninstall when LinkArch is waiting for t_notify_a
        (0.8, (QubitState.RAW, QubitState.RAW), (0, 0)),
        # uninstall when LinkArch is waiting for t_notify_b
        (1.1, (QubitState.ENTANGLED1, QubitState.RAW), (1, 0)),
        # uninstall when both qubits are owned by NetworkLayer
        (1.5, (QubitState.ENTANGLED1, QubitState.ENTANGLED1), (1, 1)),
        # uninstall when one qubit is owned by NetworkLayer
        (4.3, (QubitState.RAW, QubitState.ENTANGLED1), (2, 2)),
        (6.1, (QubitState.ENTANGLED1, QubitState.RAW), (3, 3)),
    ],
)
def test_uninstall(uninstall_t: float, qubits_state: tuple[QubitState, QubitState], n_entangle: tuple[int, int]):
    """
    Verify proper cleanups when the path is uninstalled in various steps.
    """
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.3, "link_arch": LinkArchAlways(LinkArchSr())},
        cchannel_args={"delay": 0.3},
        memory_args={"t_cohere": 2.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 6.5, install_to=(log, net))

    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)
    manage_active_channel(0.1, nl1, nl2)
    manage_active_channel(uninstall_t, nl1, nl2, start=False)

    def assert_states(expected: tuple[QubitState, QubitState]) -> None:
        mq1 = nl1.memory.read(0, must=True)
        mq2 = nl2.memory.read(0, must=True)
        actual = mq1[0].state, mq2[0].state
        assert actual == expected

    def check_states(t: float, expected0: QubitState, expected1: QubitState) -> None:
        if t > uninstall_t:
            return
        event = func_to_event(simulator.time(sec=t), assert_states, (expected0, expected1))
        event.priority = 10000
        simulator.add_event(event)

    check_states(uninstall_t, *qubits_state)

    # t=0.1, path is installed with n1 as primary and n2 as secondary
    # t=0.1, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(0.1, QubitState.ACTIVE, QubitState.RAW)
    # t=0.4, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(0.4, QubitState.ACTIVE, QubitState.RESERVED)
    # t=0.7, n1 receives RESERVE_RES, sets qubit state to RESERVED; n2 emits photon
    check_states(0.7, QubitState.RESERVED, QubitState.RESERVED)
    # t=1.0, n1 absorbs photon, sends S-R heralding, sets qubit state to ENTANGLED
    check_states(1.0, QubitState.ENTANGLED1, QubitState.RESERVED)
    # t=1.3, n2 receives S-R heralding, sets qubit state to ENTANGLED
    check_states(1.3, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    nl1.release_after.append(None)
    nl2.release_after.append(None)
    # t=2.7, EPR decoheres, qubit states are set to RELEASE then RAW
    # t=2.7, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(2.7, QubitState.ACTIVE, QubitState.RAW)
    # t=3.0, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(3.0, QubitState.ACTIVE, QubitState.RESERVED)
    # t=3.3, n1 receives RESERVE_RES, sets qubit state to RESERVED; n2 emits photon
    check_states(3.3, QubitState.RESERVED, QubitState.RESERVED)
    # t=3.6, n1 absorbs photon, sends S-R heralding, sets qubit state to ENTANGLED
    check_states(3.6, QubitState.ENTANGLED1, QubitState.RESERVED)
    # t=3.9, n2 receives S-R heralding, sets qubit state to ENTANGLED
    check_states(3.9, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    nl1.release_after.append(4.2 - 3.6)
    # t=4.2, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    check_states(4.2, QubitState.ACTIVE, QubitState.ENTANGLED1)
    # t=4.4, n2 receives RESERVE_REQ but has no available qubit
    nl2.release_after.append(4.7 - 3.9)
    # t=4.7, n2 sets qubit state to RELEASE then RAW then RESERVED, sends RESERVE_RES
    check_states(4.7, QubitState.ACTIVE, QubitState.RESERVED)
    # t=5.0, n1 receives RESERVE_RES, sets qubit state to RESERVED; n2 emits photon
    check_states(5.0, QubitState.RESERVED, QubitState.RESERVED)
    # t=5.3, n1 absorbs photon, sends S-R heralding, sets qubit state to ENTANGLED
    check_states(5.3, QubitState.ENTANGLED1, QubitState.RESERVED)
    # t=5.6, n2 receives S-R heralding, sets qubit state to ENTANGLED
    check_states(5.6, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    nl2.release_after.append(6.0 - 5.6)
    # t=6.0, n2 sets qubit state to RELEASE then RAW but has no pending request
    check_states(6.0, QubitState.ENTANGLED1, QubitState.RAW)
    nl1.release_after.append(6.4 - 5.3)
    # t=6.4, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    # t=6.5, scenario ends

    simulator.run()

    assert (len(nl1.entangle), len(nl2.entangle)) == n_entangle


def test_skip_ahead():
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"length": 100},
        cchannel_args={"length": 100},
        memory_args={"t_cohere": 1.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1)

    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    ll1, ll2 = (node.get_app(LinkLayer) for node in net.nodes)
    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)
    manage_active_channel(0.5, nl1, nl2)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)

    assert len(nl1.entangle) == len(nl2.entangle) > 0
    for t1, t2 in zip(nl1.entangle, nl2.entangle, strict=True):
        assert t1 == pytest.approx(t2, abs=1e-3)

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert 0 <= ll_cnt_agg.decoh_ratio <= 1


def test_timing_mode_sync():
    topo = CustomTopology(
        {
            "qnodes": [
                {"name": "n0"},
                {"name": "n1"},
                {"name": "n2"},
                {"name": "n3"},
            ],
            "qchannels": [
                {"node1": "n0", "node2": "n1", "parameters": {"delay": 0.2, "link_arch": LinkArchAlways(LinkArchDimBk())}},
                {"node1": "n2", "node2": "n3", "parameters": {"delay": 0.1, "link_arch": LinkArchAlways(LinkArchDimBk())}},
            ],
        },
        nodes_apps=[NetworkLayer(), LinkLayer()],
        memory_args={"t_cohere": 10.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, timing=TimingModeSync(t_ext=0.6, t_int=0.4))
    net.build_route()

    simulator = Simulator(0.0, 6.1, install_to=(log, net))

    nl0, nl1, nl2, nl3 = (node.get_app(NetworkLayer) for node in net.nodes)
    manage_active_channel(0.1, nl0, nl1)
    manage_active_channel(0.1, nl2, nl3)  # insertion_count=1, start entanglements
    manage_active_channel(1.1, nl2, nl3)  # insertion_count=2, no change
    manage_active_channel(4.1, nl2, nl3, start=False)  # insertion_count=1, no change
    manage_active_channel(5.9, nl2, nl3, start=False)  # insertion_count=0, stop entanglements

    simulator.run()

    for nl in nl0, nl1:
        print(nl.node.name, nl.entangle, nl.decohere)
        # τ=0.2 for the channel between n0 and n1.
        # Entanglement (including reservation) requires 4τ i.e. 0.8 seconds but the EXTERNAL phase
        # has only 0.6 seconds, so that no entanglement could complete on this channel.
        assert len(nl.entangle) == 0
        assert len(nl.decohere) == 0

    for nl in nl2, nl3:
        print(nl.node.name, nl.entangle, nl.decohere)
        # τ=0.1 for the channel between n2 and n3.
        # Entanglement (including reservation) requires 4τ i.e. 0.4 seconds.
        # No entanglement occurs in the first EXTERNAL phase window, because reservations are only initiated
        # at the start of each EXTERNAL phase window, not when ManageActiveChannels arrives.
        # The uninstall_path stop event takes effect for the EXTERNAL phase window starting at t=6.0.
        assert [t_notify for t_notify, _ in nl.entangle] == pytest.approx([1.4, 2.4, 3.4, 4.4, 5.4], abs=1e-3)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(nl.decohere) == 0
    QuantumMemory.check_leaks([nl2.node, nl3.node])
