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


import itertools
from collections import defaultdict
from collections.abc import Sequence
from typing import Final, override

from mqns.network.fw import MultiplexingVector, MultiplexingVectorInput, PathInstructions, RoutingController
from mqns.network.network import Request, RequestActiveEvent, RequestInactiveEvent, RequestState
from mqns.network.proactive.ctrl_ru import PathDemands, ResourceUtilization
from mqns.network.proactive.mux_input import MuxSchemeInput, mux_scheme_is_buffer_space
from mqns.simulator import event_handler
from mqns.utils import unwrap_cast


class ProactiveRoutingController(RoutingController):
    """
    Centralized control plane for proactive routing.
    Works with ``ProactiveForwarder`` on quantum nodes.

    This controller is compatible with both SYNC and SYNC timing modes.
    It can automatically pick up requests added through ``QuantumNetwork``.
    """

    ru: ResourceUtilization | None = None
    """
    Controller's view of network-wide resource utilization.

    This is created only if the network uses buffer-space multiplexing scheme.
    """

    drq: dict[int, Request]
    """
    Deferred request queue.

    These requests are in their active periods but not accepted due to insufficient resources.
    They could be accepted when resources become available.

    Note: this is a ``dict`` (keyed by ``req_id``) rather than a ``deque``.
    A ``dict`` preserves insertion order, but allows deleting an arbitrary entry.
    """

    drq_cap: Final[int]
    """
    Capacity of deferred request queue.
    """

    def __init__(self, *, mux: MuxSchemeInput, drq_cap=0):
        """
        Args:
            mux: Multiplexing scheme used in the network.
            drq_cap: Deferred request queue capacity, 0 disables queuing.
        """
        super().__init__()
        self._is_buffer_space = mux_scheme_is_buffer_space(mux)
        self.drq_cap = drq_cap
        self.drq = {}

    @override
    def install(self, node) -> None:
        super().install(node)
        if self._is_buffer_space:
            self.ru = ResourceUtilization(self.net)

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req
        req.ctrl_data = ctrl_data = _CtrlData()

        # Construct RoutingPath and compute paths.
        rp = self.prepare_path(req)
        req_id = rp.req_id
        ctrl_data.insts = insts = rp.compute_paths(self.route_ctx)

        # If the network does not use buffer-space multiplexing scheme, accept the request.
        if not self.ru:
            return self._req_accept(req)

        # If the network uses buffer-space multiplexing scheme:
        # - Populate MultiplexingVector in each PathInstructions.
        # - Gather resource demands from MultiplexingVector.
        self._populate_mv(insts, rp.bufferspace_mv)
        ctrl_data.demands = demands = self.ru.gather_demands(insts)

        # If there are sufficient resources, accept the request.
        if not demands.find_violations():
            return self._req_accept(req)

        # If the deferred request queue is full, reject the request.
        if (drq_len := len(self.drq)) >= self.drq_cap:
            req.state = RequestState.REJECTED
            self.log_debug(
                "REQ_REJECT req_id=%s reason=no-resource drq-len=%s | %s | %s | %s", req_id, drq_len, self.ru, demands, insts
            )
        # Otherwise, enqueue the request.
        else:
            req.state = RequestState.DEFERRED
            self.drq[req_id] = req
            self.log_debug(
                "REQ_DEFER req_id=%s reason=no-resource drq-len=%s | %s | %s | %s", req_id, drq_len, self.ru, demands, insts
            )

    def _req_accept(self, req: Request) -> None:
        ctrl_data: _CtrlData = req.ctrl_data

        # If the network uses buffer-space multiplexing scheme, commit the resources.
        if self.ru:
            ctrl_data.demands.commit()

        # Send PATH_INSERT.
        ctrl_data.iph = self.path_insert(req.req_id, ctrl_data.insts, epr_count=req.epr_count)
        ctrl_data.insts.clear()  # no longer needed

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        req = event.req

        # If the request was rejected, there's nothing to uninstall.
        if req.state is RequestState.REJECTED:
            return

        # If the request was deferred and is still not accepted, set it to rejected.
        if req.state is RequestState.DEFERRED:
            req.state = RequestState.REJECTED
            del self.drq[req.req_id]
            self.log_debug("REQ_REJECT req_id=%s reason=deferred-until-inactive", req.req_id)
            return

        # Send PATH_DELETE.
        ctrl_data: _CtrlData = req.ctrl_data
        self.path_delete(ctrl_data.iph)

        # If the network uses buffer-space multiplexing scheme, release the committed resources after fib_erase_delay.
        if self.ru:
            ctrl_data.demands.release(cb_after=self._drq_retry)

    def _drq_retry(self) -> None:
        """
        Retry deferred requests, called after any resource is freed.
        """
        accepted: list[int] = []
        for req_id, req in self.drq.items():
            ctrl_data: _CtrlData = req.ctrl_data
            if not ctrl_data.demands.find_violations():
                req.state = RequestState.ACTIVE
                accepted.append(req_id)
                self._req_accept(req)
        for req_id in accepted:
            del self.drq[req_id]

    def _populate_mv(self, insts: Sequence[PathInstructions], input: MultiplexingVectorInput) -> None:
        if len(insts) > 1 and input == "max":
            self._mv_divide(insts)
            return

        for inst in insts:
            inst["bufferspace_mv"] = self._mv_individual(inst["route"], input)

    def _mv_divide(self, insts: Sequence[PathInstructions]) -> None:
        # For bufferspace_mv == "max", count how many paths share the same quantum channel.
        # Note that this only counts among paths generated by this RoutingPath and would not
        # consider other RoutingPath(s) in the network.
        qchannel_use_count = defaultdict[tuple[str, str], int](lambda: 0)
        for inst in insts:
            for a, b in itertools.pairwise(inst["route"]):
                qchannel_use_count[(a, b) if a < b else (b, a)] += 1

        # Equally divide the channel capacity by how many paths share the channel.
        for inst in insts:
            mv: MultiplexingVector = []
            for a, b in itertools.pairwise(inst["route"]):
                shared = qchannel_use_count[(a, b) if a < b else (b, a)]
                for node, neighbor in (a, b), (b, a):
                    ncu = unwrap_cast(self.ru).nodes[node].channels[neighbor]
                    mv.append(ncu.n_qubits // shared)
            inst["bufferspace_mv"] = mv

    def _mv_individual(self, route: list[str], input: MultiplexingVectorInput) -> MultiplexingVector:
        if input in ("max", 0):
            mv: MultiplexingVector = []
            for a, b in itertools.pairwise(route):
                for node, neighbor in (a, b), (b, a):
                    ncu = unwrap_cast(self.ru).nodes[node].channels[neighbor]
                    mv.append(ncu.n_qubits)
            return mv

        if isinstance(input, int):
            assert input > 0
            return [input, input] * (len(route) - 1)

        return input


class _CtrlData:
    insts: list[PathInstructions]
    demands: PathDemands
    iph: RoutingController.InsertedPathHandle
