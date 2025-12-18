"""
Test suite for swapping in proactive forwarding, integrated with LinkLayer.
"""

from copy import deepcopy

import numpy as np
import pytest

from mqns.entity.qchannel import LinkArchDimBk
from mqns.network.network import TimingMode, TimingModeAsync, TimingModeSync
from mqns.network.proactive import (
    CutoffSchemeWaitTime,
    MuxSchemeDynamicEpr,
    ProactiveForwarder,
    ProactiveRoutingController,
    QubitAllocationType,
    RoutingPathMulti,
    RoutingPathSingle,
    RoutingPathStatic,
)

from .proactive_common import (
    CheckUnchanged,
    build_dumbbell_network,
    build_linear_network,
    build_rect_network,
    check_e2e_consumed,
    dflt_qchannel_args,
    install_path,
)


def test_no_swap():
    """Test isolated links mode where swapping is disabled."""
    net, simulator = build_linear_network(3)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[0, 0, 0]))
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)
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

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3"], swap=[1, 0, 1]))
    f3.cnt.enable_collect_all()
    simulator.run()

    for fw in (f1, f2, f3):
        print(fw.own.name, fw.cnt)
    assert f1.cnt.consumed_fidelity_values is None
    print(np.histogram(f3.cnt.consumed_fidelity_values or [], bins=4))

    # entanglements at n2 are immediately eligible because there's no purification
    assert f1.cnt.n_entg + f3.cnt.n_entg == f2.cnt.n_entg == f2.cnt.n_eligible == pytest.approx(8000, abs=5)
    # only eligible qubits may be swapped at n2, with 50% success rate
    assert 0 < f2.cnt.n_swapped <= f2.cnt.n_eligible * 0.7
    # no swapping is expected at n1 and n3
    assert f1.cnt.n_swapped == f3.cnt.n_swapped == 0
    # successful swap at n2 should make the qubit eligible at n1 and n3, but allow small loss at end of simulation
    assert f2.cnt.n_swapped >= f1.cnt.n_eligible >= f2.cnt.n_swapped - 2
    # consumptions are expected at n1 and n3;
    # consumed entanglements should have expected fidelity worse than initial fidelity
    check_e2e_consumed(f1, f3, n_swaps=f2.cnt.n_swapped, swap_balanced=True, f_min=0.78, f_max=0.84)
    # no consumption is expected at n2
    assert f2.cnt.n_consumed == 0


def test_swap_2():
    """Test 2-repeater parallel swapping."""
    net, simulator = build_linear_network(4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3", "n4"], swap=[1, 0, 0, 1]))
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

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
    # but there would be losses because parallel swaps require coincidence;
    # consumed entanglements should have expected fidelity worse than initial fidelity
    check_e2e_consumed(f1, f4, n_min=600, n_max=min(f2.cnt.n_swapped, f3.cnt.n_swapped) - 1, f_min=0.71, f_max=0.77)


def run_swap_3(timing: TimingMode):
    net, simulator = build_linear_network(5, timing=timing, end_time=20.0)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)

    install_path(ctrl, RoutingPathStatic(["n1", "n2", "n3", "n4", "n5"], swap=[2, 0, 1, 0, 2]), t_uninstall=9.999999)
    with CheckUnchanged(simulator, 10.0, 20.0, lambda: (f1.cnt.n_consumed, f5.cnt.n_consumed, f3.cnt.n_swapped)):
        simulator.run()

    for fw in (f1, f2, f3, f4, f5):
        print(fw.own.name, fw.cnt)
        assert fw.cnt.n_swapped_p == 0

    return f1, f2, f3, f4, f5


def test_swap_3_async():
    """Test 3-repeater sequential swapping, using ASYNC timing mode."""
    f1, f2, f3, f4, f5 = run_swap_3(TimingModeAsync())

    assert 2400 < f1.cnt.n_entg < 2800
    assert 4800 < f2.cnt.n_entg < 5600
    assert 4800 < f3.cnt.n_entg < 5600
    assert 4800 < f4.cnt.n_entg < 5600
    assert 2400 < f5.cnt.n_entg < 2800
    assert 1000 < f2.cnt.n_swapped < 1600
    assert 1000 < f4.cnt.n_swapped < 1600
    assert 500 < f3.cnt.n_swapped < 800
    check_e2e_consumed(f1, f5, n_swaps=f3.cnt.n_swapped, swap_balanced=True)


def test_swap_3_sync():
    """Test 3-repeater sequential swapping, using SYNC timing mode."""
    f1, f2, f3, f4, f5 = run_swap_3(
        TimingModeSync(t_ext=0.008, t_int=0.002),  # 0.01 seconds per time slot, 1000 total time slots
    )

    # first time slot is spent in processing routing commands; entanglements begin at the second time slot
    assert f1.cnt.n_entg == f5.cnt.n_entg == 999
    assert f2.cnt.n_entg == f3.cnt.n_entg == f4.cnt.n_entg == 1998
    assert 440 < f2.cnt.n_swapped < 560
    assert 440 < f4.cnt.n_swapped < 560
    assert 100 < f3.cnt.n_swapped < 160
    # final INTERNAL phase has sufficient time for consumption, can't lose any at the end
    check_e2e_consumed(f1, f5, n_swaps=f3.cnt.n_swapped, swap_balanced=True)


def test_swap_mr():
    """Test swapping over multiple requests."""
    net, simulator = build_dumbbell_network(qchannel_capacity=8, mux=MuxSchemeDynamicEpr(), end_time=90.0)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    # n4-n2-n1-n3-n6
    install_path(ctrl, RoutingPathSingle("n4", "n6", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"), t_install=10)
    # n5-n2-n1-n3-n7
    install_path(ctrl, RoutingPathSingle("n5", "n7", qubit_allocation=QubitAllocationType.DISABLED, swap="l2r"), t_uninstall=80)

    with (
        CheckUnchanged(simulator, 0, 9, lambda: (f4.cnt.n_entg, f4.cnt.n_consumed)),
        CheckUnchanged(simulator, 81, 90, lambda: (f5.cnt.n_entg, f7.cnt.n_consumed)),
    ):
        simulator.run()

    for fw in (f4, f6, f5, f7):
        print(fw.own.name, fw.cnt)

    # some end-to-end entanglements should be consumed at n4 and n6
    check_e2e_consumed(f4, f6, n_min=1, capacity=8)
    # some end-to-end entanglements should be consumed at n5 and n7
    check_e2e_consumed(f5, f7, n_min=1, capacity=8)


def test_swap_mp():
    """Test swapping over multiple paths."""
    net, simulator = build_rect_network(qchannel_capacity=4)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)

    # n1-n2-n4 and n1-n3-n4
    install_path(ctrl, RoutingPathMulti("n1", "n4", swap="swap_1"))
    simulator.run()

    for fw in (f1, f2, f3, f4):
        print(fw.own.name, fw.cnt)

    # both paths are used
    assert f2.cnt.n_swapped > 4000
    assert f3.cnt.n_swapped > 4000
    # swapped EPRs are consumed, capacity=8 is twice of qchannel_capacity because there are two paths
    check_e2e_consumed(f1, f4, n_swaps=f2.cnt.n_swapped + f3.cnt.n_swapped, swap_balanced=True, capacity=8)


def test_cutoff_waittime():
    """Test 5-repeater swapping with wait-time cutoff."""
    qchannel_args = deepcopy(dflt_qchannel_args)
    qchannel_args["link_arch"] = LinkArchDimBk()
    net, simulator = build_linear_network(7, qchannel_capacity=1, qchannel_args=qchannel_args, end_time=300)
    ctrl = net.get_controller().get_app(ProactiveRoutingController)
    f1 = net.get_node("n1").get_app(ProactiveForwarder)
    f2 = net.get_node("n2").get_app(ProactiveForwarder)
    f3 = net.get_node("n3").get_app(ProactiveForwarder)
    f4 = net.get_node("n4").get_app(ProactiveForwarder)
    f5 = net.get_node("n5").get_app(ProactiveForwarder)
    f6 = net.get_node("n6").get_app(ProactiveForwarder)
    f7 = net.get_node("n7").get_app(ProactiveForwarder)

    for fw in (f2, f3, f4, f5, f6):
        CutoffSchemeWaitTime.of(fw).cnt.enable_collect_all()

    install_path(ctrl, RoutingPathSingle("n1", "n7", swap=[3, 0, 1, 0, 2, 0, 3], swap_cutoff=[0.5] * 7))
    simulator.run()

    for fw in (f1, f2, f3, f4, f5, f6, f7):
        print(fw.own.name, fw.cnt)

    assert f1.cnt.n_cutoff[0] == 0
    assert f7.cnt.n_cutoff[0] == 0
    assert f2.cnt.n_cutoff[1] == 0
    assert f4.cnt.n_cutoff[1] == 0
    assert f6.cnt.n_cutoff[1] == 0
    assert f2.cnt.n_cutoff[0] + f4.cnt.n_cutoff[0] + f6.cnt.n_cutoff[0] > 0
    assert f1.cnt.n_cutoff[1] + f3.cnt.n_cutoff[1] + f5.cnt.n_cutoff[1] + f7.cnt.n_cutoff[1] > 0

    for fw in (f2, f3, f4, f5, f6):
        cutoff = CutoffSchemeWaitTime.of(fw)
        print(np.histogram(cutoff.cnt.wait_values or [], bins=4))
        assert fw.cnt.n_eligible / 2 >= len(cutoff.cnt.wait_values or []) >= fw.cnt.n_swapped
