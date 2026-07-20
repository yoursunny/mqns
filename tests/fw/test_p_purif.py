"""
Test suite for ProactiveForwarder focused on purification.
"""

from collections.abc import Iterable, Mapping

import pytest

from mqns.entity.memory import QuantumMemory
from mqns.network.fw import RoutingPathStatic
from mqns.network.proactive import ProactiveForwarder
from mqns.network.protocol.consumer import RequestCounters
from mqns.utils import rng

from .fw_common import (
    build_linear_network,
    check_fw_counters,
    install_path,
    print_node_counters,
    provide_entanglements,
)


def force_purify_outcome(monkeypatch: pytest.MonkeyPatch, *success: bool):
    l = list(success)

    def new_random() -> float:
        nonlocal l
        this_success, *l = l
        return 0.0 if this_success else 1.0

    monkeypatch.setattr(rng, "random", new_random)


@pytest.mark.parametrize(
    ("n_rounds", "purif_success", "n_purif"),
    [
        # 1 round, success.
        (1, [1], [1]),
        # 1 round, failure.
        (1, [0], []),
        # 2 rounds, success.
        (2, [1] * 3, [2, 1]),
        # 2 rounds, round-0 success, round-1 failure.
        (2, [1, 1, 0], [2]),
        # 2 rounds, first round-0 success, second round-0 failure, round-1 not attempted.
        (2, [1, 0], [1]),
        # 2 rounds, first round-0 failure, second round-0 success, round-1 not attempted.
        (2, [0, 1], [1]),
        # 3 rounds, success.
        (3, [1] * 7, [4, 2, 1]),
        # 6 rounds, success.
        (6, [1] * 63, [32, 16, 8, 4, 2, 1]),
    ],
)
def test_link_rounds(monkeypatch: pytest.MonkeyPatch, n_rounds: int, purif_success: list[int], n_purif: list[int]):
    """Test multi-round purification on a single link with various purification outcomes."""
    n_etg: int = 2**n_rounds
    net, simulator = build_linear_network(2, ch_capacity=n_etg, fw={"p_swap": 0.0})
    fwA = net.get_node("A").get_app(ProactiveForwarder)
    fwB = net.get_node("B").get_app(ProactiveForwarder)

    rp = install_path(net, RoutingPathStatic("AB", swap=[0, 0], purif={"A-B": n_rounds}))
    provide_entanglements(*((1.001 + i / 1000, fwA, fwB) for i in range(n_etg)))
    force_purify_outcome(monkeypatch, *(True if i > 0 else False for i in purif_success))
    simulator.run()
    print_node_counters(net)

    assert fwA.cnt.n_purif == n_purif == fwB.cnt.n_purif
    n_eligible = 0 if len(n_purif) < n_rounds else n_purif[-1]
    check_fw_counters(
        net,
        n_entg=(n_etg, n_etg),
        n_eligible=(n_eligible, n_eligible),
    )
    assert RequestCounters.of(net, rp).n_consumed == n_eligible


xfail_uninstall = pytest.mark.xfail(reason="UNINSTALL_PATH cleanup not implemented", strict=True)


@pytest.mark.parametrize(
    ("t_uninstall", "etg_sec", "cutoff", "purif"),
    [
        # 1. t=1.0010, A-B.0 arrives.
        # 2. t=1.0020, A-B.1 arrives, B sends PURIF_SOLICIT, which would arrive at 1.0025.
        # 3. t=1.0022, path is uninstalled.
        pytest.param(1.0022, ([1.0000, 1.0010], []), -1, {"A-B": 1}, id="purif-solicit", marks=xfail_uninstall),
        # 1. t=1.0010, A-B.0 arrives.
        # 2. t=1.0020, A-B.1 arrives, B sends PURIF_SOLICIT.
        # 3. t=1.0025, A receives PURIF_SOLICIT and sends PURIF_RESPONSE, which would arrive at 1.0030.
        # 4. t=1.0027, path is uninstalled.
        pytest.param(1.0027, ([1.0000, 1.0010], []), -1, {"A-B": 1}, id="purif-response", marks=xfail_uninstall),
        # 1. t=1.0010, A-B.0 arrives, B-C arrives.
        # 2. t=1.0020, A-B.1 arrives, B sends PURIF_SOLICIT.
        # 3. t=1.0025, A receives PURIF_SOLICIT and sends PURIF_RESPONSE.
        # 4. t=1.0030, B receives PURIF_RESPONSE and starts swapping.
        # 5. t=1.2030, B finishes swapping and sends SWAP_UPDATE.
        # 6. t=1.2035, A and C process SWAP_UPDATE message.
        # 7. t=1.2037, path is uninstalled.
        pytest.param(1.2037, ([1.0000, 1.0010], [1.0000]), -1, {"A-B": 1}, id="purif-complete"),
        # 1. t=1.0010, A-B arrives, discard scheduled at 1.0050.
        # 2. t=1.0012, path is uninstalled.
        pytest.param(1.0012, ([1.0000], []), 0.004, {}, id="cutoff-waiting", marks=xfail_uninstall),
        # 1. t=1.0010, A-B arrives, discard scheduled at 1.0050.
        # 2. t=1.0050, B discards A-B and sends CUTOFF_DISCARD message to A, which would arrive at 1.0055.
        # 3. t=1.0052, path is uninstalled.
        pytest.param(1.0052, ([1.0000], []), 0.004, {}, id="cutoff-inflight", marks=xfail_uninstall),
        # 1. t=1.0010, A-B arrives, discard scheduled at 1.0050.
        # 2. t=1.0050, B discards A-B and sends CUTOFF_DISCARD message to A.
        # 3. t=1.0055, A processes CUTOFF_DISCARD MESSAGE.
        # 4. t=1.0057, path is uninstalled.
        pytest.param(1.0057, ([1.0000], []), 0.004, {}, id="cutoff-complete"),
        # 1. t=1.0010, A-B arrives.
        # 2. t=1.0110, B-C arrives, B starts swapping, which would finish at 1.2110.
        # 3. t=1.1000, path is uninstalled.
        pytest.param(1.1000, ([1.0000], [1.0100]), -1, {}, id="swap-waiting", marks=xfail_uninstall),
        # 1. t=1.0010, A-B arrives.
        # 2. t=1.0110, B-C arrives, B starts swapping.
        # 3. t=1.2110, B finishes swapping and sends SWAP_UPDATE to A and C, which would arrive at 1.2115.
        # 4. t=1.2112, path is uninstalled.
        pytest.param(1.2112, ([1.0000], [1.0100]), -1, {}, id="swap-inflight", marks=xfail_uninstall),
        # 1. t=1.0010, A-B arrives.
        # 2. t=1.0110, B-C arrives, B starts swapping.
        # 3. t=1.2110, B finishes swapping and sends SWAP_UPDATE to A and C.
        # 4. t=1.2115, A and C process SWAP_UPDATE message.
        # 4. t=1.2117, path is uninstalled.
        pytest.param(1.2117, ([1.0000], [1.0100]), -1, {}, id="swap-complete"),
    ],
)
def test_3_uninstall(
    t_uninstall: float,
    etg_sec: tuple[Iterable[float], Iterable[float]],
    cutoff: float,
    purif: Mapping[str, int],
):
    """Test UNINSTALL_PATH cleanup during cut-off/swap/purify on 3-node topology."""
    net, simulator = build_linear_network(3, t_cohere=5.0, ch_capacity=2, fw={"p_swap": 1.0, "swap_delay": 0.2}, end_time=5.0)
    fwA, fwB, fwC = (node.get_app(ProactiveForwarder) for node in net.nodes)

    install_path(net, RoutingPathStatic("ABC", swap_cutoff=[cutoff, -1], purif=purif), t_uninstall=t_uninstall)
    provide_entanglements(
        (etg_sec[0], fwA, fwB),
        (etg_sec[1], fwB, fwC),
    )

    simulator.run()
    print_node_counters(net)
    QuantumMemory.check_leaks(net.nodes, deallocated=True)


def test_4_l2r(monkeypatch: pytest.MonkeyPatch):
    """Test multi-segment purification on 4-node topology with l2r swapping order."""
    net, simulator = build_linear_network(4, ch_capacity=8, fw={"p_swap": 1.0})
    fwA, fwB, fwC, fwD = (node.get_app(ProactiveForwarder) for node in net.nodes)

    rp = install_path(
        net,
        RoutingPathStatic("ABCD", swap=[2, 0, 1, 2], purif={"A-B": 1, "B-C": 1, "C-D": 1, "A-C": 1, "A-D": 1}),
    )
    provide_entanglements(
        (1.001, fwA, fwB),  # \
        (1.002, fwA, fwB),  # -+ A-B purif_rounds=1 \
        (1.003, fwB, fwC),  # \                      \
        (1.004, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0
        (1.005, fwA, fwB),  # \                             \
        (1.006, fwA, fwB),  # -+ A-B purif_rounds=1 \        + A-C purif_rounds=1
        (1.007, fwB, fwC),  # \                      \      /              \
        (1.008, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0  + A-D purif_rounds=0
        (1.009, fwC, fwD),  # \                                            /      \
        (1.010, fwC, fwD),  # -+ C-D purif_rounds=1 ----------------------/        \
        (1.011, fwA, fwB),  # \                                                     |
        (1.012, fwA, fwB),  # -+ A-B purif_rounds=1 \                               |
        (1.013, fwB, fwC),  # \                      \                              +-- A-D purif_rounds=1
        (1.014, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0          |
        (1.015, fwA, fwB),  # \                             \                       |
        (1.016, fwA, fwB),  # -+ A-B purif_rounds=1 \        + A-C purif_rounds=1  /
        (1.017, fwB, fwC),  # \                      \      /              \      /
        (1.018, fwB, fwC),  # -+ B-C purif_rounds=1 --+ A-C purif_rounds=0  + A-D purif_rounds=0
        (1.019, fwC, fwD),  # \                                            /
        (1.020, fwC, fwD),  # -+ C-D purif_rounds=1 ----------------------/
    )
    force_purify_outcome(monkeypatch, *[True] * 19)
    simulator.run()
    print_node_counters(net)

    assert fwA.cnt.n_purif == [4 + 2 + 1]  # 4 with fwB, 2 with fwC, 1 with fwD
    assert fwB.cnt.n_purif == [4 + 4]  # 4 with fwA, 4 with fwC
    assert fwC.cnt.n_purif == [4 + 2 + 2]  # 4 with fwB, 2 with fwD, 2 with fwA
    assert fwD.cnt.n_purif == [2 + 1]  # 2 with fwC, 1 with fwA

    check_fw_counters(
        net,
        # An entanglement becomes eligible if it completes all purification and the node has lower/equal swap rank.
        # This differs from .cnt.n_purif[0], which does not consider the node's swap rank.
        # fwA: 1 with fwD
        # fwB: 4 with fwA, 4 with fwC
        # fwC: 2 with fwD, 2 with fwA
        # fwD: 1 with fwA
        n_eligible=(1, 4 + 4, 2 + 2, 1),
    )
    assert RequestCounters.of(net, rp).n_consumed == 1
