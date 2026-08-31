import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from typing import Final, Literal, TypedDict, Unpack, override

from mqns.network.fw.message import MultiplexingVector, PathInstructions
from mqns.network.fw.swap_sequence import SwapSequenceInput, parse_swap_sequence
from mqns.network.network import QuantumNetwork
from mqns.simulator import Time
from mqns.utils import log

type MultiplexingVectorInput = Literal["auto", "none", "max"] | int | MultiplexingVector
"""
Buffer-space multiplexing vector or how to generate them.

* "auto": Equivalent to "max" if the network uses buffer-space multiplexing scheme, otherwise "none".
* "none": Network is not using buffer-space multiplexing scheme.
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
    bufferspace_mv: MultiplexingVectorInput
    """Buffer-space multiplexing vector or how to generate them, defaults to "auto"."""
    swap: SwapSequenceInput
    """Swap sequence or swap policy, defaults to ASAP."""
    swap_cutoff: Sequence[float] | None
    """Swap cut-off times in seconds."""
    purif: Mapping[str, int] | None
    """Purification scheme."""


class RoutingPath(ABC):
    """
    Compute routing path(s) for installing through RoutingController.
    """

    src: Final[str]
    """
    End node name at the source (left) side.
    """

    dst: Final[str]
    """
    End node name at the destination (right) side.
    """

    req_id: int
    """
    Request identifier.
    """

    path_id: int
    """
    Path identifier for the first path.
    If there are multiple paths, subsequent paths are given consecutive ids.
    """

    bufferspace_mv: MultiplexingVectorInput
    """
    Buffer-space multiplexing vector.
    """

    swap: SwapSequenceInput
    """
    Swap sequence or swap policy.
    """

    swap_cutoff: Sequence[float] | None
    """
    Swap cut-off values in seconds.
    """

    purif: dict[str, int]
    """
    Purification scheme.
    """

    _computed_paths: list[PathInstructions] | None = None

    def __init__(self, src: str, dst: str, **kwargs: Unpack[RoutingPathInitArgs]):
        self.src = src
        self.dst = dst
        self.req_id = kwargs.get("req_id", -1)
        self.path_id = kwargs.get("path_id", -1)
        self.bufferspace_mv = kwargs.get("bufferspace_mv", "auto")
        self.swap = kwargs.get("swap") or "asap"
        self.swap_cutoff = kwargs.get("swap_cutoff")
        self.purif = dict(kwargs.get("purif") or {})

    def list_paths(self, net: QuantumNetwork, *, recompute=False) -> Sequence[PathInstructions]:
        """
        Compute and return a list of path instructions.

        Pre-conditions:

        * ``self.bufferspace_mv`` is not "auto".
        * ``self.req_id`` and ``self.path_id`` are assigned to non-negative values.

        Args:
            net: The quantum network.
                 ``net.build_route()`` must have been called prior to invoking this function.
            recompute: If False, use previously computed results if available.

        Returns:
            A list of path instructions.
        """
        assert self.req_id >= 0
        assert self.path_id >= 0
        if recompute or self._computed_paths is None:
            self._computed_paths = list(self.compute_paths(net))
        return self._computed_paths

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
        path_id: int,
        *,
        override_mv: MultiplexingVector | None = None,
    ) -> PathInstructions:
        swap = parse_swap_sequence(self.swap, route)
        inst: PathInstructions = {
            "path_id": path_id,
            "route": route,
            "swap": swap,
            "purif": self.purif,
        }

        if self.swap_cutoff is not None:
            accuracy = net.simulator.accuracy
            inst["swap_cutoff"] = [-1 if t < 0 else Time.sec_to_slot(t, accuracy) for t in self.swap_cutoff]

        mv = self._compute_mv(net, route) if override_mv is None else override_mv
        if mv is not None:
            inst["bufferspace_mv"] = mv

        return inst

    def _compute_mv(self, net: QuantumNetwork, route: Sequence[str]) -> MultiplexingVector | None:
        _ = net
        n_hops = len(route) - 1
        mv = self.bufferspace_mv

        if mv == "auto":
            raise RuntimeError("bufferspace_mv=auto must be replaced by caller")

        if mv == "none":
            return None

        if mv == "max":
            mv = 0

        if isinstance(mv, int):
            assert mv >= 0
            return [mv, mv] * n_hops

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
        yield self._make_path_instructions(net, self.route, self.path_id)


class RoutingPathSingle(RoutingPath):
    """
    Compute a single shortest path for installing through RoutingController.
    """

    @override
    def compute_paths(self, net: QuantumNetwork) -> Iterator[PathInstructions]:
        route = net.query_route(self.src, self.dst)[0]
        log.debug("ROUTING: Computed path #%s: %s", self.path_id, route)
        yield self._make_path_instructions(net, route.path, self.path_id)


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

            mv: MultiplexingVector | None = None

            if self.bufferspace_mv == "max":
                # For "max", equally divide the channel capacity by how many paths share the channel.
                mv = []
                for node_a, node_b in pairwise(route.nodes):
                    ch = net.get_qchannel(node_a.name, node_b.name)
                    shared = qchannel_use_count[ch.name]
                    assert shared > 0

                    mv += (
                        sum(1 for _ in node_a.memory.find(lambda *_: True, qchannel=ch)) // shared,
                        sum(1 for _ in node_b.memory.find(lambda *_: True, qchannel=ch)) // shared,
                    )

            yield self._make_path_instructions(net, route.path, path_id, override_mv=mv)
