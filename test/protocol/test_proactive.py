import pytest

from qns.entity.cchannel import ClassicChannelInitKwargs
from qns.entity.node import Controller
from qns.entity.qchannel import LinkArchAlways, LinkArchDimBk, QuantumChannelInitKwargs
from qns.network.network import QuantumNetwork
from qns.network.proactive import (
    LinkLayer,
    MuxScheme,
    MuxSchemeBufferSpace,
    MuxSchemeDynamicEpr,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathMulti,
    RoutingPathSingle,
    RoutingPathStatic,
)
from qns.network.proactive.message import validate_path_instructions
from qns.network.route import RouteImpl, YenRouteAlgorithm
from qns.network.topology import ClassicTopology, GridTopology, LinearTopology, Topology, TreeTopology
from qns.simulator import Simulator
from qns.utils import log

init_fidelity = 0.90

qchannel_args = QuantumChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
    link_arch=LinkArchAlways(LinkArchDimBk()),  # entanglement in 0.002 seconds
)

cchannel_args = ClassicChannelInitKwargs(
    length=100,  # delay is 0.0005 seconds
)


def build_network_finish(
    topo: Topology,
    qchannel_capacity: int,
    *,
    end_time: float,
    route: RouteImpl | None = None,
):
    topo.controller = Controller("ctrl", apps=[ProactiveRoutingController()])

    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.Follow, route=route)
    for qchannel in net.qchannels:
        qchannel.assign_memory_qubits(capacity=qchannel_capacity)
    topo.connect_controller(net.nodes)

    simulator = Simulator(0.0, end_time)
    log.install(simulator)
    net.install(simulator)

    return net, simulator


def build_linear_network(
    n_nodes: int,
    *,
    qchannel_capacity=1,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
) -> tuple[QuantumNetwork, Simulator]:
    topo = LinearTopology(
        n_nodes,
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time=end_time)


def build_dumbbell_network(
    *,
    qchannel_capacity=1,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
) -> tuple[QuantumNetwork, Simulator]:
    """
    Build the following topology:

        n4           n6
        |            |
        +n2---n1---n3+
        |            |
        n5           n7
    """
    topo = TreeTopology(
        nodes_number=7,
        children_number=2,
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 3 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time=end_time)


def build_rect_network(
    *,
    qchannel_capacity=1,
    mux: MuxScheme = MuxSchemeBufferSpace(),
    end_time=10.0,
) -> tuple[QuantumNetwork, Simulator]:
    """
    Build the following topology:

        n1---n2
        |     |
        n3---n4
    """
    topo = GridTopology(
        (2, 2),
        nodes_apps=[
            LinkLayer(init_fidelity=init_fidelity),
            ProactiveForwarder(ps=0.5, mux=mux),
        ],
        qchannel_args=qchannel_args,
        cchannel_args=cchannel_args,
        memory_args={"decoherence_rate": 1 / 5.0, "capacity": 2 * qchannel_capacity},
    )
    return build_network_finish(topo, qchannel_capacity, end_time=end_time, route=YenRouteAlgorithm(k_paths=2))


def test_path_validation():
    """Test path validation logic."""

    route3 = ["n1", "n2", "n3"]
    swap3 = [1, 0, 1]
    mv3 = [(1, 1)] * 2

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions({"req_id": 0, "route": [], "swap": [], "purif": {}})

    with pytest.raises(ValueError, match="swapping order"):
        validate_path_instructions({"req_id": 0, "route": ["n1", "n2", "n3", "n4", "n5"], "swap": [0, 0, 0], "purif": {}})

    with pytest.raises(ValueError, match="multiplexing vector"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": [(1, 1)] * 3, "purif": {}})

    with pytest.raises(ValueError, match="purif segment r1-r2"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"r1-r2": 1}})

    with pytest.raises(ValueError, match="purif segment n1-n2-n3"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n1-n2-n3": 1}})

    with pytest.raises(ValueError, match="purif segment n2-n2"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n2-n2": 1}})

    with pytest.raises(ValueError, match="purif segment n3-n1"):
        validate_path_instructions({"req_id": 0, "route": route3, "swap": swap3, "m_v": mv3, "purif": {"n3-n1": 1}})


def test_no_swap():
    """Test isolated links mode where swapping is disabled."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[0, 0, 0]))
    simulator.run()

    for fw in (f1, f2, f3):
        print((fw.own.name, fw.cnt))
        assert 0.88 <= fw.cnt.consumed_avg_fidelity <= 0.90

    assert f1.cnt.n_entg == f1.cnt.n_eligible == f1.cnt.n_consumed == pytest.approx(5000, abs=5)
    assert f2.cnt.n_entg == f2.cnt.n_eligible == f2.cnt.n_consumed == pytest.approx(10000, abs=5)
    assert f3.cnt.n_entg == f3.cnt.n_eligible == f3.cnt.n_consumed == pytest.approx(5000, abs=5)
    assert f1.cnt.n_swapped == f2.cnt.n_swapped == f3.cnt.n_swapped == 0
    assert f1.cnt.n_consumed + f3.cnt.n_consumed == f2.cnt.n_consumed


def test_swap_1():
    """Test basic 1-repeater swapping."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1]))
    simulator.run()

    for fw in (f1, f2, f3):
        print((fw.own.name, fw.cnt))

    # entanglements at n2 are immediately eligible because there's no purification
    assert f1.cnt.n_entg + f3.cnt.n_entg == f2.cnt.n_entg == f2.cnt.n_eligible == pytest.approx(8000, abs=5)
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    # no swapping is expected at n1 and n3
    assert f1.cnt.n_swapped == f3.cnt.n_swapped == 0
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow 1 lost at end of simulation
    assert f2.cnt.n_swapped >= f1.cnt.n_eligible >= f2.cnt.n_swapped - 2
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed
    # consumed entanglement should have expected fidelity worse than initial fidelity
    assert f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
    assert 0.78 <= f1.cnt.consumed_avg_fidelity <= 0.84
    # no consumption is expected at n2
    assert f2.cnt.n_consumed == 0


def test_swap_parallel():
    """Test parallel swapping."""
    net, simulator = build_linear_network(4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3", "n4"], swap=[1, 0, 0, 1]))
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print((fw.own.name, fw.cnt))

    # entanglements at n2 and n3 are immediately eligible because there's no purification
    assert f2.cnt.n_entg == f2.cnt.n_eligible > 6000
    assert f3.cnt.n_entg == f3.cnt.n_eligible > 6000
    # only eligible qubits may be swapped at n2 and n3, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    assert 0 < f3.cnt.n_swapped <= f3.cnt.n_eligible * 0.7
    # some swaps were completed in parallel
    assert f2.cnt.n_swapped_p > 0
    assert f3.cnt.n_swapped_p > 0
    # successful swap at both n2 and n3 can make the qubit eligible at n1 and n4,
    # but there would be losses because parallel swaps require coincidence
    assert min(f2.cnt.n_swapped, f3.cnt.n_swapped) > f1.cnt.n_eligible > 0
    # eligible qubits at n1 and n4 are immediately consumed
    assert f1.cnt.n_eligible == f4.cnt.n_eligible == f1.cnt.n_consumed == f4.cnt.n_consumed >= 600
    # consumed entanglement should have expected fidelity worse than initial fidelity
    assert f1.cnt.consumed_avg_fidelity == pytest.approx(f4.cnt.consumed_avg_fidelity, abs=1e-3)
    assert 0.71 <= f1.cnt.consumed_avg_fidelity <= 0.77


def test_swap_mr():
    """Test swapping over multiple requests."""
    net, simulator = build_dumbbell_network(qchannel_capacity=2, mux=MuxSchemeDynamicEpr(), end_time=60.0)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    # n4-n2-n1-n3-n6
    ctrl.install_path(RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"))
    # n5-n2-n1-n3-n7
    ctrl.install_path(RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"))
    simulator.run()

    for app in (f4, f6, f5, f7):
        print((app.own.name, app.cnt))

    # some end-to-end entanglements should be consumed at n4 and n6
    assert pytest.approx(f4.cnt.n_consumed, abs=1) == f6.cnt.n_consumed > 0
    # some end-to-end entanglements should be consumed at n5 and n7
    assert pytest.approx(f5.cnt.n_consumed, abs=1) == f7.cnt.n_consumed > 0


def test_swap_mp():
    """Test swapping over multiple paths."""
    net, simulator = build_rect_network(qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    # n1-n2-n4 and n1-n3-n4
    ctrl.install_path(RoutingPathMulti("n1", "n4", swap="swap_1"))
    simulator.run()

    for app in (f1, f2, f3, f4):
        print((app.own.name, app.cnt))

    # end-to-end entanglements are consumed at n1 and n4
    assert f1.cnt.n_consumed == f4.cnt.n_consumed > 12000
    # each swap allows one end-to-end entanglements, but some can be lost
    assert f1.cnt.n_consumed <= f2.cnt.n_swapped_s + f3.cnt.n_swapped_s < f1.cnt.n_consumed * 1.02
    # both paths are used
    assert f2.cnt.n_swapped_s > 4000
    assert f3.cnt.n_swapped_s > 4000


def test_purif_link1r():
    """Test 1-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=2)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 1, "n2-n3": 1}))
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

        # some purifications should fail
        assert app.cnt.n_purif[0] < app.cnt.n_entg * 0.8

    # entanglements at n2 are eligible after 1-round purification
    assert pytest.approx(f1.cnt.n_purif[0] + f3.cnt.n_purif[0], abs=1) == f2.cnt.n_purif[0] == f2.cnt.n_eligible
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow small loss at end of simulation
    assert f2.cnt.n_swapped >= f1.cnt.n_eligible >= f2.cnt.n_swapped - 2
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 1000
    # consumed entanglement should have expected fidelity better than swap_1
    assert f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
    assert 0.82 <= f1.cnt.consumed_avg_fidelity <= 0.88


def test_purif_link2r():
    """Test 2-round purification on each link."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n2": 2, "n2-n3": 2}))
    simulator.run()

    for app in (f1, f2, f3):
        print((app.own.name, app.cnt))

        # some purifications should fail
        assert app.cnt.n_purif[0] < app.cnt.n_entg * 0.8
        assert app.cnt.n_purif[1] < app.cnt.n_purif[0] * 0.8

    # entanglements at n2 are eligible after 2-round purification
    assert pytest.approx(f1.cnt.n_purif[0] + f3.cnt.n_purif[0], abs=1) == f2.cnt.n_purif[0]
    assert pytest.approx(f1.cnt.n_purif[1] + f3.cnt.n_purif[1], abs=1) == f2.cnt.n_purif[1] == f2.cnt.n_eligible
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow 1 lost at end of simulation
    assert f2.cnt.n_swapped >= f1.cnt.n_eligible >= f2.cnt.n_swapped - 2
    # eligible qubits at n1 and n3 are immediately consumed
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 600
    # consumed entanglement should have expected fidelity better than purif_link1r
    assert f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
    assert 0.86 <= f1.cnt.consumed_avg_fidelity <= 0.92


def test_purif_ee2r():
    """Test 2-round purification between two end nodes."""
    net, simulator = build_linear_network(3, qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    ctrl.install_path(RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1], purif={"n1-n3": 2}))
    simulator.run()

    for fw in (f1, f2, f3):
        print((fw.own.name, fw.cnt))

    # successful swap at n2 enables first round of n1-n3 purification; some purifications should fail
    assert pytest.approx(f1.cnt.n_purif[0], abs=1) == f3.cnt.n_purif[0] < f2.cnt.n_swapped * 0.6
    assert pytest.approx(f1.cnt.n_purif[1], abs=1) == f3.cnt.n_purif[1] < f3.cnt.n_purif[0] * 0.6
    # no purification may occur in n2
    assert len(f2.cnt.n_purif) == 0
    # eligible qubits at n1 and n3 are consumed after completing 2-round purification
    assert f1.cnt.n_eligible == f3.cnt.n_eligible == f1.cnt.n_consumed == f3.cnt.n_consumed >= 500
    # consumed entanglement should have expected fidelity better than swap_1
    assert f1.cnt.consumed_avg_fidelity == pytest.approx(f3.cnt.consumed_avg_fidelity, abs=1e-3)
    assert 0.85 <= f1.cnt.consumed_avg_fidelity <= 0.91
