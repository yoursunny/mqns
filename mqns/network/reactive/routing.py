import itertools
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import override

from mqns.network.fw import RoutingPath
from mqns.network.fw.message import PathInstructions, QubitKeySequence
from mqns.network.network import QuantumNetwork, Request
from mqns.network.reactive.message import LinkStateEntry


class TopoLinkState:
    """
    Topology link state -- controller's view of available EPRs in the network.
    """

    def __init__(self):
        self.d = defaultdict[tuple[str, str], deque[str]](deque)
        """
        Key: node names, sorted.
        Value: entanglement reservation keys.
        """

    def clear(self) -> None:
        self.d.clear()

    def add(self, entry: LinkStateEntry) -> None:
        """
        Save a link state entry.
        """
        n0 = entry["node"]
        n1 = entry["neighbor"]
        if n0 < n1:
            self.d[n0, n1].append(entry["qubit"])

    def try_consume(self, route: list[str]) -> list[str] | None:
        """
        Attempt to match a computed route with available entanglements.
        The entanglements are removed this table only if every link along the path has an entanglement.
        """
        link_etgs: list[deque[str]] = []

        for n0, n1 in itertools.pairwise(route):
            etgs = self.d.get((n0, n1) if n0 < n1 else (n1, n0))
            if not etgs:
                return None
            link_etgs.append(etgs)

        return [etgs.popleft() for etgs in link_etgs]


type ReactiveRoutingPathDef = tuple[list[str], QubitKeySequence]
"""
Path computed by ``ReactiveRoutingController``.

* [0]: List of nodes.
* [1]: EPR key on each link.
"""


class ReactiveRoutingPath(RoutingPath):
    """
    Store paths computed by ``ReactiveRoutingController`` for a src-dst pair.
    """

    def __init__(self, req: Request):
        super().__init__(req.src, req.dst, **req.rp_args)

        self.paths: list[ReactiveRoutingPathDef] = []
        """List of computed paths with specific EPRs."""

        # Clear unsupported fields.
        self.m_v = "none"
        self.purif = {}

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        for route, qubits in self.paths:
            inst = self._make_path_instructions(net, route)
            inst["reactive_qubits"] = qubits
            yield inst
