#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
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

from collections.abc import Iterable
from typing import cast, overload

from mqns.entity.base_channel import BaseChannel
from mqns.entity.cchannel import ClassicChannel
from mqns.entity.node import Controller, Node, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement, WernerStateEntanglement
from mqns.network.network.request import Request, RequestActiveEvent
from mqns.network.network.timing import TimingMode, TimingModeAsync
from mqns.network.route import DijkstraRouteAlgorithm, RouteAlgorithm, RouteQueryResult
from mqns.network.topology import ClassicTopology, Topology
from mqns.simulator import Simulator, Time


def _save_channel[C: BaseChannel](l: list[C], d: dict[tuple[str, str], C], ch: C) -> None:
    l.append(ch)
    if len(ch.node_list) != 2:
        return
    a, b = sorted((node.name for node in cast(list[Node], ch.node_list)))
    d[(a, b)] = ch


def _get_channel[C: BaseChannel](l: list[C], d: dict[tuple[str, str], C], q: tuple[str, ...]) -> C:
    if len(q) == 1:
        name = q[0]
        for ch in l:
            if ch.name == name:
                return ch
        raise LookupError(f"channel {name} does not exist") from None

    a, b = q
    if a > b:
        a, b = b, a

    try:
        return d[(a, b)]
    except KeyError:
        raise LookupError(f"channel between {a} and {b} does not exist") from None


class QuantumNetwork:
    """QuantumNetwork includes quantum nodes, quantum and classical channels, arranged in a given topology"""

    def __init__(
        self,
        topo: Topology | None = None,
        *,
        classic_topo: ClassicTopology | None = None,
        route: RouteAlgorithm[QNode, QuantumChannel] | None = None,
        timing: TimingMode = TimingModeAsync(),
        epr_type: type[Entanglement] = WernerStateEntanglement,
    ):
        """
        Args:
            topo: Topology builder.
            classic_topo: Classic topology parameter, passed to topology builder.
            route: Routing algorithm, defaults to dijkstra.
            timing: Network-wide application timing mode.
            epr_type: Network-wide entanglement type.
        """
        assert getattr(epr_type, "__final__", False) is True, f"entanglement type {epr_type} must be marked @final"

        self.timing = timing
        """Network-wide application timing mode."""
        self.epr_type = epr_type
        """Network-wide entanglement type."""

        self.controller: Controller | None = None
        """Controller node."""
        self.nodes: list[QNode] = []
        """List of quantum nodes."""
        self._node_by_name: dict[str, QNode] = {}
        self.qchannels: list[QuantumChannel] = []
        """List of quantum channels."""
        self._qchannel_by_ends: dict[tuple[str, str], QuantumChannel] = {}
        self.cchannels: list[ClassicChannel] = []
        """List of classic channels."""
        self._cchannel_by_ends: dict[tuple[str, str], ClassicChannel] = {}

        if topo is not None:
            self._populate_from_topo(topo, classic_topo)

        self.route: RouteAlgorithm = route or DijkstraRouteAlgorithm()
        """Routing algorithm."""

        self.requests: list[Request] = []
        """Requested end-to-end entanglements."""

    def _populate_from_topo(self, topo: Topology, classic_topo: ClassicTopology | None) -> None:
        nodes, qchannels = topo.build()
        if classic_topo is not None:
            cchannels = topo.add_cchannels(classic_topo=classic_topo, nl=nodes, ll=qchannels)
        else:
            cchannels = topo.add_cchannels()

        for node in nodes:
            self.add_node(node)
        for ch in qchannels:
            self.add_qchannel(ch)
        for ch in cchannels:
            self.add_cchannel(ch)

        if topo.controller:
            self.set_controller(topo.controller)

    def _ensure_not_installed(self) -> None:
        """
        Assert that this entity has not been installed into a simulator.
        """
        assert not hasattr(self, "simulator"), "function only available prior to self.install()"

    def install(self, simulator: Simulator) -> None:
        """
        Install all nodes (including channels, memories and applications) in this network.

        Args:
            simulator: The simulator.
        """
        self.simulator = simulator
        """Simulator instance."""

        self.all_nodes: list[Node] = []
        """A collection of quantum nodes and the controller (if present)."""
        self.all_nodes += self.nodes
        if self.controller:
            self.all_nodes.append(self.controller)

        self.timing.install(self)

        for node in self.all_nodes:
            node.install(simulator)

        for req in self.requests:
            self._install_request(req)

    def add_node(self, node: QNode) -> None:
        """
        Add a QNode into this network.
        """
        self._ensure_not_installed()
        assert node.name not in self._node_by_name, f"duplicate node name {node.name}"
        self.nodes.append(node)
        self._node_by_name[node.name] = node
        node.add_network(self)

    def get_node(self, name: str) -> QNode:
        """
        Get QNode by name.

        Raises:
            LookupError: Node does not exist.
        """
        try:
            return self._node_by_name[name]
        except KeyError:
            raise LookupError(f"node {name} does not exist") from None

    def set_controller(self, controller: Controller) -> None:
        """
        Set the controller of this network.
        """
        self._ensure_not_installed()
        self.controller = controller
        controller.add_network(self)

    def get_controller(self) -> Controller:
        """
        Get the Controller of this network.

        Raises:
            LookupError: Controller does not exist.
        """
        if self.controller is None:
            raise LookupError("network does not have a controller")
        return self.controller

    def add_qchannel(self, qchannel: QuantumChannel) -> None:
        """
        Add a QuantumChannel into this network.
        """
        self._ensure_not_installed()
        _save_channel(self.qchannels, self._qchannel_by_ends, qchannel)

    @overload
    def get_qchannel(self, name: str, /) -> QuantumChannel:
        """
        Retrieve QuantumChannel by name.

        Raises:
            LookupError: channel does not exist.
        """

    @overload
    def get_qchannel(self, a: str, b: str, /) -> QuantumChannel:
        """
        Retrieve QuantumChannel by node names.

        Raises:
            LookupError: channel does not exist.
        """

    def get_qchannel(self, *q: str) -> QuantumChannel:
        return _get_channel(self.qchannels, self._qchannel_by_ends, q)

    def add_cchannel(self, cchannel: ClassicChannel) -> None:
        """
        Add a ClassicChannel into this network.
        """
        self._ensure_not_installed()
        _save_channel(self.cchannels, self._cchannel_by_ends, cchannel)

    @overload
    def get_cchannel(self, name: str, /) -> ClassicChannel:
        """
        Retrieve ClassicalChannel by name.

        Raises:
            LookupError: channel does not exist.
        """

    @overload
    def get_cchannel(self, a: str, b: str, /) -> ClassicChannel:
        """
        Retrieve ClassicalChannel by node names.

        Raises:
            LookupError: channel does not exist.
        """

    def get_cchannel(self, *q: str) -> ClassicChannel:
        return _get_channel(self.cchannels, self._cchannel_by_ends, q)

    def build_route(self) -> None:
        """Build static route tables."""
        self.route.build(self.nodes, self.qchannels)

    def query_route(self, src: str, dst: str, /, error_on_empty=True) -> list[RouteQueryResult[QNode]]:
        """
        Query the routing algorithm.

        Args:
            src: Source node.
            dst: Destination node.
            error_on_empty: If true, raise RuntimeError if there's no route.

        Returns:
            List of route paths, sorted by priority.
        """
        routes = self.route.query(self.get_node(src), self.get_node(dst))
        if error_on_empty and not routes:
            raise RuntimeError(f"no route from {src} to {dst}")
        return routes

    @property
    def active_requests(self) -> Iterable[Request]:
        """
        List requests that are active at current timestamp.
        See ``Request.is_active()`` for the criteria of determining whether a request is active.
        """
        t = self.simulator.tc
        return (req for req in self.requests if req.is_active(t))

    def add_request(self, *reqs: Request) -> None:
        """
        Add one or more requests to the network.
        """
        self.requests.extend(reqs)

        if hasattr(self, "simulator"):
            for req in reqs:
                self._install_request(req)

    def _install_request(self, req: Request) -> None:
        req.active_since = Time.from_time_or_sec(req.active_since_input, accuracy=self.simulator.accuracy)
        req.active_until = Time.from_time_or_sec(req.active_until_input, accuracy=self.simulator.accuracy)
        del req.active_since_input
        del req.active_until_input

        if not self.controller:
            return

        t_enter = self.simulator.tc if req.active_since is Time.MIN else req.active_since
        self.simulator.add_event(RequestActiveEvent(self.controller, req, True, t=t_enter))

        if req.active_until is not Time.MAX:
            req.inactive_event = RequestActiveEvent(self.controller, req, False, t=req.active_until)
            self.simulator.add_event(req.inactive_event)
