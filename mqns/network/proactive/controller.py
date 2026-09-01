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


from typing import cast, override

from mqns.network.fw import RoutingController, RoutingPathMulti, RoutingPathSingle
from mqns.network.network import RequestActiveEvent, RequestInactiveEvent, RequestState
from mqns.network.proactive.ctrl_ru import PathDemands, ResourceUtilization
from mqns.network.proactive.mux_input import MuxSchemeInput, mux_scheme_is_buffer_space
from mqns.network.route import YenRouteAlgorithm
from mqns.simulator import event_handler
from mqns.utils import unwrap


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

    def __init__(self, *, mux: MuxSchemeInput):
        self._is_buffer_space = mux_scheme_is_buffer_space(mux)
        super().__init__(mv_auto="max" if self._is_buffer_space else "none")

    @override
    def install(self, node) -> None:
        super().install(node)
        if self._is_buffer_space:
            self.ru = ResourceUtilization(self.net)

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req

        # Construct RoutingPath for the request.
        if (rp := req.rp) is None:
            req.rp = rp = (
                RoutingPathMulti(req.src, req.dst, **req.rp_args)
                if isinstance(self.net.route, YenRouteAlgorithm)
                else RoutingPathSingle(req.src, req.dst, **req.rp_args)
            )

        self.prepare_path(rp)
        paths = rp.list_paths(self.net, recompute=True)

        # If the network uses buffer-space multiplexing scheme, gather the resource demands of the computed paths.
        if self.ru is not None:
            demands = self.ru.gather_demands(paths)

            # If there are insufficient resources, reject the request.
            if demands.has_violation:
                req.state = RequestState.REJECTED
                self.log_debug("REQ_REJECT req_id=%s reason=no-resource | %s | %s | %s", req.req_id, self.ru, demands, paths)
                return

            # If there are sufficient resources, commit these resources.
            demands.commit()
            rp.ctrl_data = demands

        # Send PATH_INSERT.
        self.install_path(rp, recompute=False, epr_count=req.epr_count)

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        req = event.req

        # If the request was rejected, there's nothing to uninstall.
        if req.state is RequestState.REJECTED:
            return

        # Send PATH_DELETE.
        rp = unwrap(req.rp)
        self.uninstall_path(rp)

        # If the network uses buffer-space multiplexing scheme, release the committed resources after fib_erase_delay.
        if self.ru is not None:
            demands = cast(PathDemands, rp.ctrl_data)
            demands.release()
