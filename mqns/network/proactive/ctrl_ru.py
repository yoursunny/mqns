import itertools
from collections.abc import Iterable, Mapping
from typing import Final

from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.message import PathInstructions
from mqns.network.network import QuantumNetwork
from mqns.network.proactive.fw_nb import to_fib_erase_delay
from mqns.simulator import Simulator, Time, func_to_event


class NodeChannelUtilization:
    """Controller's view of utilization of qubits assigned to a channel."""

    nu: Final["NodeUtilization"]
    """Node utilization reference."""

    neighbor: Final[str]
    """Neighbor name."""

    n_qubits: Final[int]
    """How many memory qubits are assigned to this channel."""

    unallocated_qubits: int
    """How many memory qubits are available for allocation to paths."""

    def __init__(self, nu: "NodeUtilization", node: QNode, ch: QuantumChannel):
        self.nu = nu
        self.neighbor = ch.find_peer(node).name
        self.n_qubits = sum(1 for _ in node.memory.find(QuantumMemory.all, qchannel=ch))
        self.unallocated_qubits = self.n_qubits

    def __repr__(self) -> str:
        return f"{self.neighbor}({self.unallocated_qubits}/{self.n_qubits})"

    @property
    def name(self) -> str:
        return f"{self.nu.name}-{self.neighbor}"


class NodeUtilization:
    """Controller's view of node resource utilization."""

    name: Final[str]
    """Node name."""

    channels: Final[dict[str, NodeChannelUtilization]]
    """Per-channel assignment."""

    t_cohere: Final[Time]
    """Memory coherence time."""

    n_qubits: Final[int]
    """How many memory qubits exist in this node."""

    unassigned_qubits: int
    """How many memory qubits are available for assignment to channels."""

    def __init__(self, node: QNode):
        self.name = node.name
        self.t_cohere = node.memory.t_cohere
        self.n_qubits = node.memory.capacity
        self.unassigned_qubits = self.n_qubits

        self.channels = {}
        for ch in node.qchannels:
            ncu = NodeChannelUtilization(self, node, ch)
            self.channels[ncu.neighbor] = ncu
            self.unassigned_qubits -= ncu.n_qubits

    def __repr__(self) -> str:
        return (
            f"{self.name}({self.unassigned_qubits}/{self.n_qubits}; "
            f"channels={[ncu.__repr__() for ncu in self.channels.values()]})"
        )


class ResourceUtilization:
    """Controller's view of network-wide resource utilization."""

    simulator: Final[Simulator]

    nodes: Final[Mapping[str, NodeUtilization]]
    """Node utilization keyed by node name."""

    def __init__(self, net: QuantumNetwork):
        self.simulator = net.simulator
        self.nodes = {node.name: NodeUtilization(node) for node in net.nodes}

    def __repr__(self) -> str:
        return f"ResourceUtilization(nodes={[nu.__repr__() for nu in self.nodes.values()]})"

    def gather_demands(self, insts: Iterable[PathInstructions]) -> "PathDemands":
        """
        Compute resource demands for one or more paths.

        If the returned ``PathDemands`` indicates no violations, call ``demands.commit()`` to commit the resources.
        """
        demands = PathDemands(self)
        for inst in insts:
            demands.add_path(inst)
        return demands


class PathDemands:
    """
    Track the resource demands for one or more paths.
    """

    def __init__(self, ru: ResourceUtilization):
        self.ru: Final = ru
        self.demands: Final[dict[NodeChannelUtilization, int]] = {}
        self.violations: Final = set[str]()
        self.max_t_cohere = Time.MIN

    def __repr__(self) -> str:
        return f"PathDemands({[f'{ncu.name}:{n}' for ncu, n in self.demands.items()]}, violations={sorted(self.violations)})"

    def add_path(self, inst: PathInstructions) -> None:
        """
        Add a path to the resource demands.
        """
        assert "bufferspace_mv" in inst
        mv = inst["bufferspace_mv"]
        for i, (a, b) in enumerate(itertools.pairwise(inst["route"])):
            for j, node, neighbor in (0, a, b), (1, b, a):
                self._add_node(node, neighbor, mv[2 * i + j])

    def _add_node(self, node: str, neighbor: str, n: int) -> None:
        ncu = self.ru.nodes[node].channels[neighbor]
        if n == 0:
            n = ncu.n_qubits

        demand = self.demands.get(ncu, 0) + n
        if demand > ncu.unallocated_qubits:
            self.violations.add(ncu.name)
        self.demands[ncu] = demand

    @property
    def has_violation(self) -> bool:
        """
        Return whether committing these paths would cause resource violations.
        """
        return len(self.violations) > 0

    def commit(self) -> None:
        """
        Commit the resources for these paths.

        Pre-conditions:

        * There is no resource violations, i.e. ``has_violation is False``.
        """
        assert not self.has_violation, "PathDemands has resource violations"
        for ncu, demand in self.demands.items():
            ncu.unallocated_qubits -= demand
            self.max_t_cohere = max(self.max_t_cohere, ncu.nu.t_cohere)

    def release(self) -> None:
        """
        Release the resources for these paths.

        The release is internally delayed by a duration equal to forwarder's FIB erase delay.

        Pre-conditions:

        * The resources was committed, i.e. ``commit()`` was called.
        """
        assert self.max_t_cohere is not Time.MIN, "PathDemands was not committed"
        simulator = self.ru.simulator
        fib_erase_delay = to_fib_erase_delay(self.max_t_cohere)
        simulator.sched(func_to_event(simulator.tc + fib_erase_delay, self._revert))

    def _revert(self) -> None:
        for ncu, demand in self.demands.items():
            ncu.unallocated_qubits += demand
