from collections import defaultdict, deque
from typing import override

import pytest

from mqns.entity.memory import MemoryDecohereEvent, MemoryQubit, PathDirection, QuantumMemory, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import LinkArchAlways, LinkArchDimBk, LinkArchSr, QuantumChannel
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.network.network import QuantumNetwork, TimingModeSync
from mqns.network.protocol.event import PathActivateEvent, PathDeactivateEvent, QubitEntangledEvent, QubitReleasedEvent
from mqns.network.protocol.link_layer import LinkLayer, LinkLayerCounters
from mqns.network.topology import ClassicTopology, CustomTopology, LinearTopology
from mqns.simulator import Simulator, event_handler, func_to_event
from mqns.utils import log, rng


@pytest.fixture(autouse=True)
def _reseed_env():
    rng.reseed("env")


class NetworkLayer(Application[QNode]):
    check_epr_creation_prior: float | None = None
    """
    If set, ``handle_entangle`` verifies that every EPR was created this duration earlier.
    """

    def __init__(self):
        super().__init__()
        self.release_after = deque[float | None]()
        """If non-empty, ``QubitReleasedEvent`` would be emitted after specified duration for the next entanglement."""
        self.entangle: list[float] = []
        """Entanglement events, each entry is event time."""
        self.path_entangle = defaultdict[int | None, list[float]](list)
        """Entanglement times per path_id."""
        self.decohere: list[float] = []
        """Decoherence events, each entry is event time."""

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        self.epr_type = self.node.network.epr_type

    @event_handler
    def handle_entangle(self, event: QubitEntangledEvent) -> None:
        mq, epr = self.memory.read(event.qubit.addr, has=self.epr_type)
        assert mq is event.qubit
        if self.check_epr_creation_prior is not None:
            t_create = epr.decohere_time - self.memory.t_cohere
            assert t_create == event.t - self.check_epr_creation_prior
        self.entangle.append(event.t.sec)
        self.path_entangle[mq.path_id].append(event.t.sec)

        try:
            release_after = self.release_after.popleft()
        except IndexError:
            return
        if release_after is not None:
            self.simulator.sched(func_to_event(event.t + release_after, self._release, mq))

    def _release(self, mq: MemoryQubit):
        self.memory.read(mq.addr, remove=True)
        mq.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, mq, t=self.simulator.tc))

    @event_handler
    def handle_decohere(self, event: MemoryDecohereEvent) -> None:
        self.decohere.append(event.t.sec)
        event.qubit.state = QubitState.RELEASE
        self.simulator.sched(QubitReleasedEvent(self.node, event.qubit, is_decoh=True, t=event.t))


def activate_path(t0: float | None, t1: float | None, src: Application[QNode], dst: Application[QNode], path_id: int | None):
    """
    Schedule ``PathActivateEvent`` and ``PathDeactivateEvent``.
    """
    simulator = src.simulator
    ch = src.node.get_qchannel(dst.node)

    if t0 is not None:
        t = simulator.time(sec=t0)
        simulator.sched(PathActivateEvent(src.node, ch, path_id, t=t, is_primary=True))
        simulator.sched(PathActivateEvent(dst.node, ch, path_id, t=t, is_primary=False))

    if t1 is not None:
        t = simulator.time(sec=t1)
        simulator.sched(PathDeactivateEvent(src.node, ch, path_id, t=t))
        simulator.sched(PathDeactivateEvent(dst.node, ch, path_id, t=t))


def force_attempts(monkeypatch: pytest.MonkeyPatch, ll: LinkLayer, /, **kwargs: list[int]):
    """
    Force entanglements to succeed at k-th attempt.

    Args:
        ll: LinkLayer instance.
        kwargs: Attempts numbers for upcoming entanglements, keyed by partner name.
    """

    def new_calc_attempts(qchannel: QuantumChannel) -> int:
        nonlocal kwargs
        partner = qchannel.find_peer(ll.node)
        l = kwargs[partner.name]
        this_attempts, *l = l
        kwargs[partner.name] = l
        return this_attempts

    monkeypatch.setattr(ll, "_calc_attempt", new_calc_attempts)


@pytest.mark.parametrize("epr_type", [WernerStateEntanglement, MixedStateEntanglement])
@pytest.mark.parametrize("path_id", [None, 7])
def test_basic(monkeypatch: pytest.MonkeyPatch, epr_type: type[Entanglement], path_id: int | None):
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.1, "link_arch": LinkArchDimBk()},
        cchannel_args={"delay": 0.1},
        memory_args={"t_cohere": 4.1},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow, epr_type=epr_type)
    net.build_route()
    net.get_qchannel("n1", "n2").assign_memory_qubits(capacity=1, path_id=path_id)

    simulator = Simulator(0.0, 20.0, install_to=(log, net))

    ll1, ll2 = (node.get_app(LinkLayer) for node in net.nodes)
    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)
    activate_path(0.5, 9.4, nl1, nl2, path_id)

    nl1.check_epr_creation_prior = 0.2
    nl2.check_epr_creation_prior = 0.2
    nl1.release_after.append(2.9)
    nl2.release_after.append(3.2)
    force_attempts(monkeypatch, ll1, n2=[1, 2, 1])

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)
        assert len(nl.entangle) == 3
        assert len(nl.decohere) == 2
        # t=0.5, path is installed with n1 as primary and n2 as secondary
        # t=0.5, n1 sends RESERVE_REQ
        # t=0.6, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=0.7, n1 receives RESERVE_RES, 1st attempt starts
        # t=0.9, entanglement established after 1 attempt, creation time was t=0.7
        assert nl.entangle[0] == pytest.approx(0.9, abs=1e-6)
        # t=3.8, n1 releases qubit and sends RESERVE_REQ
        # t=3.9, n2 receives RESERVE_REQ but has no qubit available
        # t=4.1, n2 releases qubit and sends RESERVE_RES
        # t=4.2, n1 receives RESERVE_RES, 1st attempt starts
        # t=4.4, 1st attempt fails, 2nd attempt starts
        # t=4.6, entanglement established after 2 attempts, creation time was t=4.4
        assert nl.entangle[1] == pytest.approx(4.6, abs=1e-6)
        # t=8.5, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[0] == pytest.approx(8.5, abs=1e-6)
        # t=8.5, n1 sends RESERVE_REQ
        # t=8.6, n2 receives RESERVE_REQ and sends RESERVE_RES
        # t=8.7, n1 receives RESERVE_RES, 1st attempt starts
        # t=8.9, entanglement established after 1 attempt, creation time was t=8.7
        assert nl.entangle[2] == pytest.approx(8.9, abs=1e-6)
        # t=9.4, path is deleted
        # t=12.8, qubits decohered 4.1 seconds since entanglement creation
        assert nl.decohere[1] == pytest.approx(12.8, abs=1e-6)
        # no more entanglements because the path has been deleted

    ll_cnt_agg = LinkLayerCounters.aggregate(net.nodes)
    print("ll_cnt_agg", ll_cnt_agg)
    assert ll_cnt_agg.n_etg == 3
    assert ll_cnt_agg.n_attempts == 4
    # n_decoh is only incremented on the primary node.
    # Although there are two decoherence events in the simulation, n1.n_decoh is incremented only at t=8.3.
    # For the t=12.6 event, the path is deleted, so that n1 cannot recognize itself as primary.
    assert ll_cnt_agg.n_decoh == 1
    assert ll_cnt_agg.decoh_ratio == pytest.approx(1 / 3, abs=1e-6)

    QuantumMemory.check_leaks(net.nodes)


def test_multiple_paths():
    """
    Test multiple active paths.
    """
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.005},
        cchannel_args={"delay": 0.005},
        memory_args={"capacity": 1 + 2 + 3 + 4, "t_cohere": 0.1},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    ch = net.get_qchannel("n1", "n2")
    ch.assign_memory_qubits(capacity=1 + 2 + 3 + 4)

    simulator = Simulator(0.0, 10.0, install_to=(log, net))

    ll1, ll2 = (node.get_app(LinkLayer) for node in net.nodes)
    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)

    def alloc_activate_path(src: NetworkLayer, dst: NetworkLayer, path_id: int, n: int) -> None:
        src.memory.allocate(ch, path_id, PathDirection.R, n=n)
        dst.memory.allocate(ch, path_id, PathDirection.L, n=n)
        activate_path(0.1, 9.9, src, dst, path_id)

    alloc_activate_path(nl1, nl2, 1, 1)
    alloc_activate_path(nl1, nl2, 2, 2)
    alloc_activate_path(nl1, nl2, 3, 3)
    alloc_activate_path(nl2, nl1, 4, 4)
    # Path 4 is requesting n2-n1 direction, but the channel is already activated in n1-n2 direction
    # so that path 4 would retain the existing n1-n2 direction.

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        path_entangle_cnts = [len(nl.path_entangle[path_id]) for path_id in range(1, 5)]
        print(ll.node.name, ll.cnt, f"path_entangle_cnts={path_entangle_cnts}")
        assert all(c > 0 for c in path_entangle_cnts)

    # Given the channel is operating in n1-n2 direction for all paths,
    # all attempts should be initiated by ll1.
    assert ll1.cnt.n_attempts > 0
    assert ll2.cnt.n_attempts == 0


@pytest.mark.parametrize(
    ("t_delete", "entangled_at_delete", "n_entangle"),
    [
        # don't delete, only check timeline states
        (999, (0, 0), 3),
        # ---------------- first entanglement ----------------
        # path deleted when RESERVE_REQ is in-flight
        (0.2, (0, 0), 0),
        # path deleted when RESERVE_RES is in-flight
        (0.5, (0, 0), 0),
        # path deleted when waiting for t_notify_pri
        (0.8, (0, 0), 1),
        # path deleted when waiting for t_notify_2nd
        (1.1, (1, 0), 1),
        # path deleted when both qubits are owned by NetworkLayer
        (1.5, (1, 1), 1),
        # ---------------- second entanglement ----------------
        # path deleted during first failed attempt
        (3.4, (0, 0), 1),
        # path deleted during second failed attempt
        (4.0, (0, 0), 1),
        # path deleted when nl2 owns a qubit
        (6.4, (0, 1), 2),
        # ---------------- third entanglement ----------------
        # path deleted when nl1 owns a qubit
        (8.2, (1, 0), 3),
    ],
)
def test_path_delete(
    monkeypatch: pytest.MonkeyPatch,
    t_delete: float,
    entangled_at_delete: tuple[int, int],
    n_entangle: int,
):
    """
    Verify PATH_DELETE cleanup when deletion occurs in various steps.
    """
    topo = LinearTopology(
        nodes_number=2,
        nodes_apps=[NetworkLayer(), LinkLayer()],
        qchannel_args={"delay": 0.3, "link_arch": LinkArchSr()},
        cchannel_args={"delay": 0.3},
        memory_args={"t_cohere": 2.0},
    )
    net = QuantumNetwork(topo, classic_topo=ClassicTopology.Follow)
    net.build_route()
    ch = net.get_qchannel("n1", "n2")
    ch.assign_memory_qubits(capacity=1, path_id=7)

    simulator = Simulator(0.0, 100.0, install_to=(log, net))

    ll1, ll2 = (node.get_app(LinkLayer) for node in net.nodes)
    nl1, nl2 = (node.get_app(NetworkLayer) for node in net.nodes)
    nl1.check_epr_creation_prior = 0.3
    nl2.check_epr_creation_prior = 0.6
    activate_path(0.1, t_delete, nl1, nl2, 7)
    force_attempts(monkeypatch, ll1, n2=[1, 4, 1, 10000])

    def assert_la_delays() -> None:
        la = ch.link_arch
        assert la.attempt_interval == simulator.time(sec=0.6)
        assert la.d_notify_pri == simulator.time(sec=0.3)
        assert la.d_notify_2nd == simulator.time(sec=0.6)

    simulator.sched(func_to_event(simulator.time(sec=0.2), assert_la_delays))

    def check_states(t: float, expected1: QubitState | None, expected2: QubitState | None) -> None:
        if t > t_delete:
            return

        def assert_states():
            if expected1 is not None:
                mq1, _ = nl1.memory.read(0, must=True)
                assert mq1.state is expected1
            if expected2 is not None:
                mq2, _ = nl2.memory.read(0, must=True)
                assert mq2.state is expected2

        event = func_to_event(simulator.time(sec=t), assert_states)
        event.priority = 10000
        simulator.sched(event)

    check_states(t_delete, *(QubitState.ENTANGLED1 if is_entangled else None for is_entangled in entangled_at_delete))

    expect_t_entangle1: list[float] = []
    expect_t_entangle2: list[float] = []
    # ---------------- first entanglement, k=1, decohere ----------------
    # t=0.1, path is installed with n1 as primary and n2 as secondary
    # t=0.1, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(0.1, QubitState.ACTIVE, QubitState.RAW)
    # t=0.4, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(0.4, QubitState.ACTIVE, QubitState.RESERVED)
    # t=0.7, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(0.7, QubitState.RESERVED, QubitState.RESERVED)
    # t=1.0, n1 is notified, sets qubit state to ENTANGLED
    check_states(1.0, QubitState.ENTANGLED1, QubitState.RESERVED)
    expect_t_entangle1.append(1.0)
    # t=1.3, n2 is notified, sets qubit state to ENTANGLED
    check_states(1.3, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    expect_t_entangle2.append(1.3)
    # t=2.7, EPR decoheres, qubit states are set to RELEASE then RAW
    nl1.release_after.append(None)
    nl2.release_after.append(None)
    #
    # ---------------- second entanglement, k=4, nl1 releases before nl2 ----------------
    # t=2.7, n1 sets qubit state to ACTIVE, sends RESERVE_REQ
    check_states(2.7, QubitState.ACTIVE, QubitState.RAW)
    # t=3.0, n2 receives RESERVE_REQ, sets qubit state to RESERVED, sends RESERVE_RES
    check_states(3.0, QubitState.ACTIVE, QubitState.RESERVED)
    # t=3.3, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would fail
    # t=3.9, first attempt fails, second attempt begins and it would fail
    # t=4.5, second attempt fails, third attempt begins and it would fail
    # t=5.1, third attempt fails, fourth attempt begins and it would succeed
    check_states(5.3, QubitState.RESERVED, QubitState.RESERVED)
    # t=5.4, n1 is notified, sets qubit state to ENTANGLED
    check_states(5.4, QubitState.ENTANGLED1, QubitState.RESERVED)
    expect_t_entangle1.append(5.4)
    # t=5.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(5.7, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    expect_t_entangle2.append(5.7)
    #
    # ---------------- third entanglement, k=1, nl2 releases before nl1 ----------------
    # t=6.3, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(6.3 - expect_t_entangle1[-1])
    check_states(6.3, QubitState.ACTIVE, QubitState.ENTANGLED1)
    # t=6.5, n2 receives RESERVE_REQ but has no available qubit
    # t=6.8, n2 sets qubit state to RELEASE then RAW then RESERVED, sends RESERVE_RES
    nl2.release_after.append(6.8 - expect_t_entangle2[-1])
    check_states(6.8, QubitState.ACTIVE, QubitState.RESERVED)
    # t=7.1, n1 receives RESERVE_RES, sets qubit state to RESERVED; first attempt begins and it would succeed
    check_states(7.1, QubitState.RESERVED, QubitState.RESERVED)
    # t=7.4, n1 is notified, sets qubit state to ENTANGLED
    expect_t_entangle1.append(7.4)
    check_states(7.4, QubitState.ENTANGLED1, QubitState.RESERVED)
    # t=7.7, n2 is notified, sets qubit state to ENTANGLED
    check_states(7.7, QubitState.ENTANGLED1, QubitState.ENTANGLED1)
    expect_t_entangle2.append(7.7)
    # t=8.1, n2 sets qubit state to RELEASE then RAW but has no pending request
    nl2.release_after.append(8.1 - expect_t_entangle2[-1])
    check_states(8.1, QubitState.ENTANGLED1, QubitState.RAW)
    # t=8.5, n1 sets qubit state to RELEASE then RAW then ACTIVE, sends RESERVE_REQ
    nl1.release_after.append(8.5 - expect_t_entangle1[-1])
    # t=100, scenario ends

    simulator.run()

    if t_delete < simulator.te.sec:
        assert len(ll1.channels) == 0
        assert len(ll2.channels) == 0
    assert nl1.entangle == pytest.approx(expect_t_entangle1[:n_entangle], abs=1e-6)
    assert nl2.entangle == pytest.approx(expect_t_entangle2[:n_entangle], abs=1e-6)


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
    activate_path(0.5, None, nl1, nl2, None)

    simulator.run()

    for ll, nl in (ll1, nl1), (ll2, nl2):
        print(ll.node.name, ll.cnt, nl.entangle, nl.decohere)

    assert len(nl1.entangle) == len(nl2.entangle) > 0
    for t1, t2 in zip(nl1.entangle, nl2.entangle, strict=True):
        assert t1 == pytest.approx(t2, abs=1e-6)

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
    activate_path(0.1, None, nl0, nl1, None)
    activate_path(0.1, 5.9, nl2, nl3, None)  # insertion_count=1
    activate_path(1.1, 4.1, nl2, nl3, None)  # insertion_count=2

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
        # at the start of each EXTERNAL phase window, not when PathActivateEvent arrives.
        # The PathDeactivateEvent takes effect for the EXTERNAL phase window starting at t=6.0.
        assert nl.entangle == pytest.approx([1.4, 2.4, 3.4, 4.4, 5.4], abs=1e-6)
        # All qubits are cleared at the start of each EXTERNAL phase, before memory decoherence occurs.
        # Decoherence events are not emitted for cleared qubits.
        assert len(nl.decohere) == 0
    QuantumMemory.check_leaks([nl2.node, nl3.node])
