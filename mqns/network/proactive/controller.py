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


from mqns.network.fw import QubitAllocationType, RoutingController, RoutingPath, RoutingPathMulti, RoutingPathSingle
from mqns.network.network import Request, RequestActiveEvent
from mqns.network.route import YenRouteAlgorithm
from mqns.simulator import event_handler


class ProactiveRoutingController(RoutingController):
    """
    Centralized control plane for proactive routing.
    Works with ``ProactiveForwarder`` on quantum nodes.

    This controller does not pick up requests from ``QuantumNetwork.requests`` list.
    If desired, scenario script should manually pass requests to ``ProactiveRoutingController.install_path()``.
    """

    def __init__(
        self,
        *,
        qubit_allocation: QubitAllocationType = QubitAllocationType.DISABLED,
    ):
        """
        Args:
            qubit_allocation: QubitAllocationType passed to ``RoutingPathSingle`` constructor
                              when converting from ``Request``.
        """
        super().__init__()
        self.qubit_allocation = qubit_allocation

    @event_handler
    def handle_request(self, event: RequestActiveEvent) -> None:
        req = event.req
        if event.enter:
            req.rp = self._path_from_request(req)
            self.install_path(req.rp)
        else:
            assert req.rp
            self.uninstall_path(req.rp)

    def _path_from_request(self, req: Request) -> RoutingPath:
        if req.rp:
            return req.rp
        if isinstance(self.net.route, YenRouteAlgorithm):
            return RoutingPathMulti(req.src, req.dst, **req.rp_args)
        return RoutingPathSingle(req.src, req.dst, qubit_allocation=self.qubit_allocation, **req.rp_args)
