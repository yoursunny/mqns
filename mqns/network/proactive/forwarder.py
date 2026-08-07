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

from typing import Unpack, override

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw import FibPath, FibRequest, Forwarder, ForwarderInitKwargs, ForwarderNorthbound
from mqns.network.protocol.event import PathActivateEvent, PathDeactivateEvent


class ProactiveForwarderNorthbound(ForwarderNorthbound):
    @override
    def install_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        self.simulator.add_event(
            PathActivateEvent(
                self.node,
                ch,
                self._ll_path_id(fp),
                t=self.simulator.tc,
                is_primary=dir is PathDirection.R,
            )
        )

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = dir
        self.simulator.add_event(
            PathDeactivateEvent(
                self.node,
                ch,
                self._ll_path_id(fp),
                t=self.simulator.tc,
            )
        )

    def _ll_path_id(self, fp: FibPath) -> int | None:
        return fp.path_id if self.mux.qubit_has_path_id() else None


class ProactiveForwarder(Forwarder):
    """
    ProactiveForwarder is the forwarder of QNodes and receives routing instructions from the controller.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    nb: ProactiveForwarderNorthbound
    """Northbound interface to communicate with the ProactiveRoutingController."""

    def __init__(self, **kwargs: Unpack[ForwarderInitKwargs]):
        super().__init__(**kwargs)
        self.nb = ProactiveForwarderNorthbound()

    @override
    def install(self, node):
        super().install(node)
        self.nb.install(self)

    @override
    def request_reached_epr_count(self, fr: FibRequest) -> None:
        self.nb.send_reach_epr_count(fr)
