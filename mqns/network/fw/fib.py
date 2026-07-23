#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, final

from mqns.network.fw.message import SwapSequence
from mqns.simulator import Time

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


@final
class FibEntry:
    """FIB entry."""

    __slots__ = ("req_id", "path_id", "active_until", "route", "own_idx", "swap", "swap_cutoff", "purif", "sg")

    req_id: Final[int]
    """Request identifier, identifies source-destination pair."""
    path_id: Final[int]
    """Path identifier, identifies end-to-end path."""

    active_until: Time
    """
    Active period upper bound (exclusive), ``Time.MAX`` means no restriction.
    This field becomes valid when the request is added to a network and a simulator is installed into the network.
    """

    route: Final[Sequence[str]]
    """List of nodes traversed by the path."""
    own_idx: Final[int]
    """Index of own node within the route."""
    swap: Final[SwapSequence]
    """Swap sequence."""
    swap_cutoff: Final[Sequence[Time | None]]
    """Swap cutoff times."""
    purif: Final[Mapping[str, int]]
    """Purification scheme."""
    sg: Final["FibSwapGroup|None"]
    """Swap group that this node belongs to; None for end-node or swap-disabled."""

    def __init__(
        self,
        *,
        req_id: int,
        path_id: int,
        route: Sequence[str],
        own_idx: int,
        swap: SwapSequence,
        swap_cutoff: Sequence[Time | None],
        purif: Mapping[str, int],
    ):
        self.req_id = req_id
        self.path_id = path_id
        self.active_until = Time.MAX
        self.route = route
        self.own_idx = own_idx
        self.swap = swap
        self.swap_cutoff = swap_cutoff
        self.purif = purif
        self.sg = None if self.own_swap_rank == self.swap[0] else FibSwapGroup(self)

    @property
    def own_swap_rank(self) -> int:
        return self.swap[self.own_idx]

    @property
    def is_swap_disabled(self) -> bool:
        """
        Determine whether swapping has been disabled.

        To disable swapping, set SwapSequence to a list of zeros.

        When swapping is disabled, the forwarder will consume entanglement upon completing purification,
        without attempting entanglement swapping.
        """
        return self.swap[0] == 0 == self.swap[-1]

    def is_active(self, t: Time) -> bool:
        """
        Determine if the time point ``t`` is within the FIB entry's active period.
        """
        return t < self.active_until

    def find_index_and_swap_rank(self, node_name: str) -> tuple[int, int]:
        """
        Determine the swapping rank of a node.

        Args:
            node_name: a node name that exists in route.

        Returns:
            [0]: The node index in the route.
            [1]: Swapping rank of the node, explained in ``PathInstructions``.

        Raises:
            LookupError: Node does not exist in route.
        """
        idx = self.route.index(node_name)
        return idx, self.swap[idx]


@final
class FibSwapGroup:
    """
    FibSwapGroup provides topological information to determine the heralding directions within
    a FIB route that may contain parallel swapping instructions.

    Each node within a FIB route, excluding the first and last end nodes, belongs to a swap group.
    The swap group of node *N* is identified with these steps:

    1. From ``entry.route``, discard nodes with a lower rank than node *N*.
    2. Find the longest continuous segment of nodes that contains *N*, where every node has the same rank as *N*.

    The left/right neighbor of a swap group is a node that has a higher rank and not part of the swap group.
    The leftmost/rightmost node of the swap group is responsible for heralding the left/right neighbor,
    if that neighbor would not be heralded by the opposite neighbor.
    """

    __slots__ = ("path_id", "rank", "nodes", "l_neigh", "r_neigh", "dir", "own_idx")

    path_id: Final[int]
    """FIB entry path ID."""

    rank: Final[int]
    """Rank of all nodes in this swap group."""

    nodes: Final[Sequence[str]]
    """Nodes in this swap group."""

    l_neigh: Final[str]
    """Left neighbor, not part of this swap group."""

    r_neigh: Final[str]
    """Right neighbor, not part of this swap group."""

    dir: Final[Literal["l", "b", "r"]]
    """
    Heralding direction when there is no swap failure.

    * b: Bidirectional -- Both neighbors have the same rank, or the segment requires purification.
    * l: Leftward -- Left neighbor has lower rank than right neighbor.
    * r: Rightward -- Right neighbor has lower rank than left neighbor.
    """

    own_idx: Final[int]
    """Index of own node within ``nodes``."""

    def __init__(self, entry: FibEntry):
        route_len = len(entry.route)
        if entry.own_idx in (0, route_len - 1):
            raise ValueError("FibSwapGroup is undefined for end nodes")

        self.path_id = entry.path_id
        self.rank = entry.own_swap_rank

        nodes = [entry.route[entry.own_idx]]
        self.l_neigh, l_rank = self._extend_1d(entry, nodes, range(entry.own_idx - 1, -1, -1))
        self.own_idx = len(nodes) - 1
        nodes.reverse()
        self.r_neigh, r_rank = self._extend_1d(entry, nodes, range(entry.own_idx + 1, route_len))
        self.nodes = nodes

        if l_rank == r_rank or entry.purif.get(f"{self.l_neigh}-{self.r_neigh}", 0) > 0:
            self.dir = "b"
        elif l_rank < r_rank:
            self.dir = "l"
        else:
            self.dir = "r"

    def _extend_1d(self, entry: FibEntry, nodes: list[str], index_range: Iterable[int]) -> tuple[str, int]:
        for i in index_range:
            node_rank = entry.swap[i]
            if node_rank > self.rank:
                return entry.route[i], entry.swap[i]
            if node_rank == self.rank:
                nodes.append(entry.route[i])
        raise ValueError(f"FibSwapGroup cannot find boundary in {entry.swap} from node index {entry.own_idx}")

    @property
    def l_most(self) -> bool:
        """Determine if own node is the leftmost node in the group."""
        return self.own_idx == 0

    @property
    def r_most(self) -> bool:
        """Determine if own node is the rightmost node in the group."""
        return self.own_idx == len(self.nodes) - 1

    @property
    def own_node(self) -> str:
        """Own node name."""
        return self.nodes[self.own_idx]

    @property
    def l_adj(self) -> str:
        """
        Left adjacent node, either same-ranked peer or higher-ranked neighbor.

        Own node should herald this node if allowed by ``dir`` or when there is a failure.
        """
        return self.l_neigh if self.l_most else self.nodes[self.own_idx - 1]

    @property
    def r_adj(self) -> str:
        """
        Right adjacent node, either same-ranked peer or higher-ranked neighbor.

        Own node should herald this node if allowed by ``dir`` or when there is a failure.
        """
        return self.r_neigh if self.r_most else self.nodes[self.own_idx + 1]

    def __repr__(self) -> str:
        tokens = [
            "FibSwapGroup(",
            self.l_neigh,
            "<" if self.dir in ("l", "b") else "~",
            "-".join(f"[{n}]" if i == self.own_idx else n for (i, n) in enumerate(self.nodes)),
            ">" if self.dir in ("b", "r") else "~",
            self.r_neigh,
            f", rank={self.rank})",
        ]
        return "".join(tokens)


class FibRequestGroup:
    """FIB information grouped by req_id."""

    def __init__(self, entry: FibEntry):
        """Construct from first FIB entry."""
        self.req_id = entry.req_id
        self.src = entry.route[0]
        self.dst = entry.route[-1]
        self.path_ids = {entry.path_id}

    def add(self, entry: FibEntry) -> None:
        """Check consistency and save FIB entry."""
        assert self.req_id == entry.req_id
        assert self.src == entry.route[0]
        assert self.dst == entry.route[-1]
        self.path_ids.add(entry.path_id)

    def remove(self, entry: FibEntry) -> bool:
        """
        Remove FIB entry.

        Returns:
            Whether the group has become empty.
        """
        assert self.req_id == entry.req_id
        self.path_ids.remove(entry.path_id)
        return len(self.path_ids) == 0


class Fib:
    def __init__(self):
        self.table: dict[int, FibEntry] = {}
        """
        FIB table.
        Key is path_id.
        Value is FIB entry.
        """
        self.by_req_id: dict[int, FibRequestGroup] = {}
        """
        Lookup table indexed by req_id.
        Key is req_id.
        Value contains aggregated information.
        """

    def install(self, fw: "Forwarder") -> None:
        self.simulator = fw.simulator

    def get(self, path_id: int) -> FibEntry:
        """
        Retrieve an entry by path_id.

        Raises:
            LookupError: Entry not found.
        """
        try:
            return self.table[path_id]
        except KeyError:
            raise LookupError(f"FIB entry not found for path_id={path_id}") from None

    def insert_or_replace(self, entry: FibEntry):
        """
        Insert an entry or replace entry with same path_id.
        """
        self.erase(entry.path_id)
        self.table[entry.path_id] = entry

        rg = self.by_req_id.get(entry.req_id)
        if rg:
            rg.add(entry)
        else:
            rg = FibRequestGroup(entry)
            self.by_req_id[rg.req_id] = rg

    def erase(self, path_id: int):
        """
        Remove an entry from the table.

        Nonexistent entry is silent ignored.
        """
        try:
            entry = self.table.pop(path_id)
        except KeyError:
            return

        rg = self.by_req_id[entry.req_id]
        if rg.remove(entry):
            del self.by_req_id[entry.req_id]

    def find_request(
        self,
        predicate: Callable[[FibRequestGroup], bool],
        *,
        has_active=False,
    ) -> Iterator[FibRequestGroup]:
        """
        List ``FibRequestGroup`` satisfying a predicate.

        Args:
            predicate: Function to determine the condition.
            has_active: Also require at least one FIB entry to be within the active period.
        """
        for rg in self.by_req_id.values():
            if predicate(rg) and ((not has_active) or self._request_has_active(rg)):
                yield rg

    def _request_has_active(self, rg: FibRequestGroup) -> bool:
        now = self.simulator.tc
        return any(self.table[path_id].is_active(now) for path_id in rg.path_ids)

    def __repr__(self):
        """Return a string representation of the forwarding table."""
        return "\n".join(
            f"Path ID: {path_id}, Request ID: {entry.req_id}, Path: {entry.route}, "
            f"Swap Sequence: {entry.swap}, Purification: {entry.purif}"
            for path_id, entry in self.table.items()
        )
