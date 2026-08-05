import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from typing import Literal, TypedDict, Unpack, override

from mqns.network.fw.message import MultiplexingVector, PathInstructions, validate_path_instructions
from mqns.network.fw.swap_sequence import SwapSequenceInput, parse_swap_sequence
from mqns.network.network import QuantumNetwork
from mqns.simulator import Time
from mqns.utils import log

type MultiplexingVectorInput = Literal["auto", "none", "max"] | int | MultiplexingVector
"""
Buffer-space multiplexing vector or how to generate them.

* "auto": Equivalent to "max" if the network uses buffer-space multiplexing scheme, otherwise "none".
* "none": No m_v, for use with statistical or dynamic EPR multiplexing schemes.
* "max": Allocate the maximum quantity of qubits per quantum channel, depending on channel capacity.
  * If multiple ``RoutingPath`` shares one channel, this would likely cause a conflict.
  * In ``RoutingPathMulti``, if the same channel is shared by multiple paths generated
    by the same ``RoutingPathMulti``, the channel capacity is equally divided among them.
* Zero: Allocate the maximum quantity of qubits per quantum channel, depending on channel capacity.
  * If multiple ``RoutingPath`` or multiple paths from a ``RoutingPathMulti`` shares one channel,
    this would likely cause a conflict.
* Positive integer: Allocate specific number of qubits per quantum channel.
* ``MultiplexingVector``: Use the pre-defined multiplexing vector, which must match route length.
"""


class RoutingPathInitArgs(TypedDict, total=False):
    req_id: int
    """Request identifier, defaults to auto-assignment."""
    path_id: int
    """Path identifier for the first path, defaults to auto-assignment."""
    swap: SwapSequenceInput
    """Swap sequence or swap policy, defaults to ASAP."""
    swap_cutoff: Sequence[float] | None
    """Swap cut-off times in seconds."""
    m_v: MultiplexingVectorInput
    """Multiplexing vector."""
    purif: Mapping[str, int] | None
    """Purification scheme."""


class RoutingPath(ABC):
    """
    Compute routing path(s) for installing through RoutingController.
    """

    def __init__(self, src: str, dst: str, **kwargs: Unpack[RoutingPathInitArgs]):
        self.src = src
        """
        Source node name.
        """
        self.dst = dst
        """
        Destination node name.
        """
        self.req_id = kwargs.get("req_id", -1)
        """
        Request identifier.

        If negative, the controller will assign the next unused value before calling ``compute_paths``.
        """
        self.path_id = kwargs.get("path_id", -1)
        """
        Path identifier for the first path.

        If negative, the controller will assign the next unused value before calling ``compute_paths``.

        When ``compute_paths`` yields multiple paths, this is the path_id on the first path,
        while subsequent paths are given consecutive path_ids.
        """
        self.swap: SwapSequenceInput = kwargs.get("swap") or "asap"
        self.swap_cutoff = kwargs.get("swap_cutoff")
        self.m_v = kwargs.get("m_v", "auto")
        self.purif = dict(kwargs.get("purif") or {})

    @abstractmethod
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        """
        Compute and yield one or more path instructions.

        Args:
            net: The quantum network.
                 ``net.build_route()`` must have been called prior to invoking this function.

        Returns:
            A generator of path instructions.
            The ``path_id`` field shall be overwritten by the caller.
        """

    def _make_path_instructions(
        self,
        net: QuantumNetwork,
        route: list[str],
        *,
        m_v: MultiplexingVector | None = None,
    ) -> PathInstructions:
        swap = parse_swap_sequence(self.swap, route)
        inst: PathInstructions = {
            "path_id": -1,
            "route": route,
            "swap": swap,
            "purif": self.purif,
        }

        if self.swap_cutoff is not None:
            accuracy = net.simulator.accuracy
            inst["swap_cutoff"] = [-1 if t < 0 else Time.sec_to_time_slot(t, accuracy) for t in self.swap_cutoff]

        if m_v is None:
            m_v = self._compute_mv(net, route)
        if m_v is not None:
            inst["m_v"] = m_v

        validate_path_instructions(inst)
        return inst

    def _compute_mv(self, net: QuantumNetwork, route: Sequence[str]) -> MultiplexingVector | None:
        _ = net
        n_hops = len(route) - 1
        mv = self.m_v

        if mv == "auto":
            raise RuntimeError("m_v=auto must be replaced by caller")

        if mv == "none":
            return None

        if mv == "max":
            mv = 0

        if isinstance(mv, int):
            assert mv >= 0
            return [(mv, mv)] * n_hops

        return mv


class RoutingPathStatic(RoutingPath):
    """
    Define a static routing path for installing through RoutingController.
    """

    def __init__(
        self,
        route: Sequence[str],
        **kwargs: Unpack[RoutingPathInitArgs],
    ):
        super().__init__(route[0], route[-1], **kwargs)
        self.route = list(route)

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        yield self._make_path_instructions(net, self.route)


class RoutingPathSingle(RoutingPath):
    """
    Compute a single shortest path for installing through RoutingController.
    """

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        route = net.query_route(self.src, self.dst)[0]
        log.debug("ROUTING: Computed path #%s: %s", self.path_id, route)
        yield self._make_path_instructions(net, route.path)


class RoutingPathMulti(RoutingPath):
    """
    Compute multiple shortest paths for installing through RoutingController.

    This should be used with YenRouteAlgorithm in the QuantumNetwork.
    The number of paths for each request is determined by the routing algorithm.
    """

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        # Compute shortest paths.
        # Number of paths is configured in the routing algorithm.
        routes = net.query_route(self.src, self.dst)

        # Count how many paths share the same quantum channel.
        # Note that this only counts among paths generated by this RoutingPathMulti and would not
        # consider other RoutingPath(s) in the network.
        qchannel_use_count = defaultdict[str, int](lambda: 0)
        for route in routes:
            for name_a, name_b in itertools.pairwise(route.path):
                ch = net.get_qchannel(name_a, name_b)
                qchannel_use_count[ch.name] += 1

        for path_id, route in enumerate(routes, start=self.path_id):
            log.debug("ROUTING: Computed path #%s: %s", path_id, route)

            m_v: MultiplexingVector | None = None

            if self.m_v == "max":
                # For m_v="max", equally divide the channel capacity by how many paths share the channel.
                m_v = []
                for node_a, node_b in pairwise(route.nodes):
                    ch = net.get_qchannel(node_a.name, node_b.name)
                    shared = qchannel_use_count[ch.name]
                    assert shared > 0

                    m_v.append(
                        (
                            sum(1 for _ in node_a.memory.find(lambda *_: True, qchannel=ch)) // shared,
                            sum(1 for _ in node_b.memory.find(lambda *_: True, qchannel=ch)) // shared,
                        )
                    )

            yield self._make_path_instructions(net, route.path, m_v=m_v)
