from collections.abc import Sequence
from typing import Literal, cast

from mqns.entity.qchannel import LinkArch, LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual, LinkArchSim, LinkArchSr
from mqns.models.epr import Entanglement, MixedStateEntanglement, WernerStateEntanglement
from mqns.models.error.input import ErrorModelInputLength
from mqns.network.network import QuantumNetwork
from mqns.network.proactive import LinkLayer, ProactiveForwarder, ProactiveRoutingController, RoutingPathSingle
from mqns.network.topology.customtopo import CustomTopology, Topo, TopoCChannel, TopoController, TopoQChannel, TopoQNode


def _broadcast[T](name: str, input: T | Sequence[T], n: int) -> Sequence[T]:
    if isinstance(input, Sequence) and not isinstance(input, (str, bytes, bytearray, memoryview)):
        assert len(input) == n, f"{name} must have {n} items"
        return input
    return [cast(T, input)] * n


def _split_channel_capacity(item: int | tuple[int, int]) -> tuple[int, int]:
    return (item, item) if isinstance(item, int) else item


CTRL_DELAY = 5e-06
"""
Delay of the classic channels between the controller and each QNode, in seconds.

In most examples, the overall simulation duration is increased by this value,
so that the QNodes can perform entanglement forwarding for the full intended duration.
"""

type EprTypeLiteral = Literal["W", "M"]
EPR_TYPE_MAP: dict[EprTypeLiteral, type[Entanglement]] = {
    "W": WernerStateEntanglement,
    "M": MixedStateEntanglement,
}

type LinkArchLiteral = Literal["DIM-BK", "DIM-BK-SeQUeNCe", "DIM-dual", "SR", "SIM"]
LINK_ARCH_MAP: dict[LinkArchLiteral, type[LinkArch]] = {
    "DIM-BK": LinkArchDimBk,
    "DIM-BK-SeQUeNCe": LinkArchDimBkSeq,
    "DIM-dual": LinkArchDimDual,
    "SR": LinkArchSr,
    "SIM": LinkArchSim,
}
type LinkArchDef = LinkArch | type[LinkArch] | LinkArchLiteral


def build_network(
    *,
    epr_type: EprTypeLiteral = "W",
    nodes: int | Sequence[str],
    mem_capacity: int | Sequence[int] | None = None,
    t_cohere: float = 0.02,
    channel_length: float | Sequence[float],
    channel_capacity: int | Sequence[int | tuple[int, int]] = 1,
    fiber_alpha: float = 0.2,
    fiber_error: ErrorModelInputLength = "DEPOLAR:0.01",
    link_arch: LinkArchDef | Sequence[LinkArchDef] = LinkArchDimBkSeq,
    entg_attempt_rate: float = 50e6,
    init_fidelity: float | None = 0.99,
    eta_d: float = 0.95,
    eta_s: float = 0.95,
    frequency: float = 1e6,
    p_swap: float = 0.5,
    swap: list[int] | str,
    swap_cutoff: list[float] | None = None,
) -> QuantumNetwork:
    """
    Build a linear topology consisting of zero or more repeaters.

    Args:
        # QuantumNetwork
        epr_type: Entanglement model, "W" for Werner state, "M" for mixed (Bell-diagonal) state.
        nodes: Number of nodes or list of node names.
        # QuantumMemory
        mem_capacity: Number of memory qubits per node.
        t_cohere: Memory coherence time in seconds.
        # QuantumChannel
        channel_length: Lengths of qchannels between adjacent nodes.
        channel_capacity: (left, right) qubit allocation per qchannel.
        fiber_alpha: Fiber loss in dB/km, determines success probability.
        fiber_rate: Fiber decoherence rate in km^{-1}, determines qualify of entangled state.
        link_arch: Link architecture per qchannel.
        # LinkLayer
        entg_attempt_rate: Maximum entanglement attempts per second.
        init_fidelity: Fidelity of generated entangled pairs.
            If ``None``, determine with error models in link architecture.
        eta_d: Detector efficiency.
        eta_s: Source efficiency.
        frequency: Entanglement source frequency.
        # ProactiveForwarder
        p_swap: probability of successful entanglement swapping.
        # ProactiveRoutingController
        swap: predefined or explicitly specified swapping order.
        swap_cutoff: cutoff times.
    """
    if isinstance(nodes, int):
        assert nodes >= 2, "at least two nodes"
        nodes = [f"R{i}" for i in range(nodes)]
        nodes[0] = "S"
        nodes[-1] = "D"

    n_nodes = len(nodes)
    assert n_nodes >= 2, "at least two nodes"
    n_links = n_nodes - 1

    channel_length = _broadcast("channel_length", channel_length, n_links)
    channel_capacity = _broadcast("channel_capacity", channel_capacity, n_links)
    link_arch = _broadcast("link_arch", link_arch, n_links)

    if mem_capacity is None:
        mem_capacity = [0]
        for caps in channel_capacity:
            capL, capR = _split_channel_capacity(caps)
            mem_capacity[-1] += capL
            mem_capacity.append(capR)
    else:
        mem_capacity = _broadcast("mem_capacity", mem_capacity, n_nodes)

    qnodes: list[TopoQNode] = []
    for name, mem_capacity in zip(nodes, mem_capacity):
        qnodes.append(
            {
                "name": name,
                "memory": {
                    "t_cohere": t_cohere,
                    "capacity": mem_capacity,
                },
            }
        )

    qchannels: list[TopoQChannel] = []
    cchannels: list[TopoCChannel] = []
    for i, (length, caps, la0) in enumerate(zip(channel_length, channel_capacity, link_arch)):
        node1, node2 = nodes[i], nodes[i + 1]
        cap1, cap2 = _split_channel_capacity(caps)
        la = LINK_ARCH_MAP.get(cast(LinkArchLiteral, la0), cast(LinkArch | type[LinkArch], la0))
        la = la() if callable(la) else la
        qchannels.append(
            {
                "node1": node1,
                "node2": node2,
                "capacity1": cap1,
                "capacity2": cap2,
                "parameters": {
                    "length": length,
                    "alpha": fiber_alpha,
                    "transfer_error": fiber_error,
                    "link_arch": la,
                },
            }
        )
        cchannels.append({"node1": node1, "node2": node2, "parameters": {"length": length}})

    path = RoutingPathSingle("S", "D", swap=swap, swap_cutoff=swap_cutoff)
    controller: TopoController = {
        "name": "ctrl",
        "apps": [ProactiveRoutingController(path)],
    }

    topo = CustomTopology(
        Topo(qnodes=qnodes, qchannels=qchannels, cchannels=cchannels, controller=controller),
        nodes_apps=[
            LinkLayer(
                attempt_rate=entg_attempt_rate,
                init_fidelity=init_fidelity,
                eta_d=eta_d,
                eta_s=eta_s,
                frequency=frequency,
            ),
            ProactiveForwarder(ps=p_swap),
        ],
    )
    net = QuantumNetwork(
        topo,
        epr_type=EPR_TYPE_MAP[epr_type],
    )
    topo.connect_controller(net.nodes, delay=CTRL_DELAY)
    return net
