import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, Literal, Protocol, TypedDict, Unpack, cast, override

from mqns.entity.memory import QuantumMemory
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.message import MultiplexingVector, PathInstructions
from mqns.network.fw.swap_sequence import SwapSequenceInput, parse_swap_sequence
from mqns.network.route import RouteQueryResult
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


class ComputeRoutesContext(Protocol):
    """
    Contextual information for ``RoutingPath.compute_routes()``.
    """

    @property
    def time_accuracy(self) -> int: ...

    def get_qchannel(self, a: str, b: str, /) -> QuantumChannel: ...

    def query_route(self, src: str, dst: str, /) -> Sequence[RouteQueryResult[QNode]]: ...

    def choose_ll_dir(self, a: str, b: str, /) -> Literal["R", "L"]:
        """
        Determine LinkLayer direction of a channel between ``a`` and ``b``.

        Returns:
            "R" makes ``a`` primary; "L" makes ``b`` primary.
        """
        ...


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

    ctrl_data: Any
    """
    Arbitrary data used by the controller.
    """

    def __init__(self, src: str, dst: str, **kwargs: Unpack[RoutingPathInitArgs]):
        self.src = src
        self.dst = dst
        self.req_id = kwargs.get("req_id", -1)
        self.path_id = kwargs.get("path_id", -1)
        self.bufferspace_mv = kwargs.get("bufferspace_mv", "auto")
        self.swap = kwargs.get("swap") or "asap"
        self.swap_cutoff = kwargs.get("swap_cutoff")
        self.purif = dict(kwargs.get("purif") or {})

    def list_paths(self, ctx: ComputeRoutesContext, *, recompute: bool) -> Sequence[PathInstructions]:
        """
        Compute and return a list of path instructions.

        Pre-conditions:

        * ``self.bufferspace_mv`` is not "auto".
        * ``self.req_id`` and ``self.path_id`` are assigned to non-negative values.

        Args:
            recompute: If False, use previously computed results if available.

        Returns:
            A list of path instructions.
        """
        assert self.req_id >= 0
        assert self.path_id >= 0
        if recompute or self._computed_paths is None:
            self._computed_paths = list(self.compute_paths(ctx))
        return self._computed_paths

    @abstractmethod
    def compute_paths(self, ctx: ComputeRoutesContext) -> Iterator[PathInstructions]:
        """
        Compute and yield one or more path instructions.

        Returns:
            A generator of path instructions.
        """

    def _make_inst(
        self,
        ctx: ComputeRoutesContext,
        path_id: int,
        route: list[str],
    ) -> PathInstructions:
        swap = parse_swap_sequence(self.swap, route)
        inst: PathInstructions = {
            "path_id": path_id,
            "route": route,
            "ll_dir": "".join(ctx.choose_ll_dir(a, b) for a, b in itertools.pairwise(route)),
            "swap": swap,
            "purif": self.purif,
        }

        if self.swap_cutoff is not None:
            inst["swap_cutoff"] = [-1 if t < 0 else Time.sec_to_slot(t, ctx.time_accuracy) for t in self.swap_cutoff]

        return inst

    def _compute_mv(self, route: Sequence[str]) -> MultiplexingVector | None:
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
    Define static routing path(s).
    """

    def __init__(
        self,
        route: Sequence[str],
        *addl_routes: Sequence[str],
        **kwargs: Unpack[RoutingPathInitArgs],
    ):
        super().__init__(route[0], route[-1], **kwargs)
        self.routes = [list(route)]
        for rt in addl_routes:
            route = list(rt)
            assert route[0] == self.src
            assert route[-1] == self.dst
            self.routes.append(route)

    @override
    def compute_paths(self, ctx: ComputeRoutesContext) -> Iterator[PathInstructions]:
        for path_id, route in enumerate(self.routes, start=self.path_id):
            inst = self._make_inst(ctx, path_id, route)
            if mv := self._compute_mv(route):
                inst["bufferspace_mv"] = mv
            yield inst


class RoutingPathSingle(RoutingPath):
    """
    Compute a single shortest path.
    """

    @override
    def compute_paths(self, ctx: ComputeRoutesContext) -> Iterator[PathInstructions]:
        route = ctx.query_route(self.src, self.dst)[0]
        log.debug("ROUTING: Computed path #%s: %s", self.path_id, route)
        inst = self._make_inst(ctx, self.path_id, route.path)
        if mv := self._compute_mv(route.path):
            inst["bufferspace_mv"] = mv
        yield inst


class RoutingPathMulti(RoutingPath):
    """
    Compute multiple shortest paths.

    This should be used with YenRouteAlgorithm in the QuantumNetwork.
    The quantity of paths is determined by the routing algorithm.
    """

    @override
    def compute_paths(self, ctx: ComputeRoutesContext) -> Iterator[PathInstructions]:
        # Compute shortest paths.
        # Number of paths is configured in the routing algorithm.
        routes = ctx.query_route(self.src, self.dst)

        # Count how many paths share the same quantum channel.
        # Note that this only counts among paths generated by this RoutingPathMulti and would not
        # consider other RoutingPath(s) in the network.
        qchannel_use_count = defaultdict[QuantumChannel, int](lambda: 0)
        for route in routes:
            for name_a, name_b in itertools.pairwise(route.path):
                ch = ctx.get_qchannel(name_a, name_b)
                qchannel_use_count[ch] += 1

        for path_id, route in enumerate(routes, start=self.path_id):
            log.debug("ROUTING: Computed path #%s: %s", path_id, route)
            inst = self._make_inst(ctx, path_id, route.path)

            if self.bufferspace_mv == "max":
                inst["bufferspace_mv"] = self._compute_mv_max(ctx, route, qchannel_use_count)
            elif mv := self._compute_mv(route.path):
                inst["bufferspace_mv"] = mv

            yield inst

    def _compute_mv_max(
        self,
        ctx: ComputeRoutesContext,
        route: RouteQueryResult[QNode],
        qchannel_use_count: Mapping[QuantumChannel, int],
    ) -> MultiplexingVector:
        # Equally divide the channel capacity by how many paths share the channel.
        mv: MultiplexingVector = []
        for node_a, node_b in pairwise(route.nodes):
            ch = ctx.get_qchannel(node_a.name, node_b.name)
            shared = cast(int, qchannel_use_count.get(ch))
            mv += (
                sum(1 for _ in node_a.memory.find(QuantumMemory.predicate_all, qchannel=ch)) // shared,
                sum(1 for _ in node_b.memory.find(QuantumMemory.predicate_all, qchannel=ch)) // shared,
            )
        return mv
