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


from mqns.network.fw import RoutingController, RoutingPathMulti, RoutingPathSingle
from mqns.network.network import RequestActiveEvent, RequestInactiveEvent
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

    @event_handler
    def handle_request_active(self, event: RequestActiveEvent) -> None:
        req = event.req
        if req.rp is None:
            req.rp = (
                RoutingPathMulti(req.src, req.dst, **req.rp_args)
                if isinstance(self.net.route, YenRouteAlgorithm)
                else RoutingPathSingle(req.src, req.dst, **req.rp_args)
            )
        self.install_path(req.rp)

    @event_handler
    def handle_request_inactive(self, event: RequestInactiveEvent) -> None:
        req = event.req
        self.uninstall_path(unwrap(req.rp))
