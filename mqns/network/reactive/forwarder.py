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
from mqns.network.fw import FibEntry, Forwarder, ForwarderInitKwargs, ForwarderNorthbound
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.network.protocol.event import PathActivateEvent
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg


class ReactiveForwarderNorthbound(ForwarderNorthbound):
    @override
    def install_path_adj(self, fib_entry: FibEntry, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = dir, ch
        if not self.node.timing.is_routing():
            self.log_warning(
                "received INSTALL_PATH message for path %s outside of ROUTING phase; t_rtg is too short?", fib_entry.path_id
            )

    @override
    def uninstall_path_adj(self, fib_entry: FibEntry, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = fib_entry, dir, ch
        raise ValueError(f"{self} should not receive UNINSTALL_PATH command")

    def send_link_state(self):
        """
        Send link state message to controller. Assumes direct connection to controller.
        """
        link_states: list[LinkStateEntry] = []
        for event in self.fw.waiting_etg:
            assert event.qubit.key is not None
            link_states.append({"node": event.node.name, "neighbor": event.neighbor.name, "qubit": event.qubit.key})

        if len(link_states) == 0:
            self.log_debug("no link_state to send")
            return
        else:
            self.log_debug("send link_state for %s etg qubits", len(self.fw.waiting_etg))

        msg: LinkStateMsg = {
            "cmd": "LS",
            "ls": link_states,
        }
        self.send_ctrl(msg)


class ReactiveForwarder(Forwarder):
    """
    ReactiveForwarder is the forwarder of QNodes. It continuously generates EPRs on all links,
    then sends EPR link state to the controller to receive routing instructions.
    Works only in Synchronous timing mode.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    nb: ReactiveForwarderNorthbound
    """Northbound interface to communicate with the ReactiveRoutingController."""

    def __init__(self, **kwargs: Unpack[ForwarderInitKwargs]):
        super().__init__(**kwargs)
        self.nb = ReactiveForwarderNorthbound()

    @override
    def install(self, node):
        super().install(node)
        self.nb.install(self)

        # Qchannel activation is called before simulation starts, but EPR generation starts at t=0
        self.activate_qchannels()

    def activate_qchannels(self):
        for ch in self.node.qchannels:
            is_primary = ch.node_list[0] is self.node
            self.simulator.add_event(PathActivateEvent(self.node, ch, None, t=self.simulator.tc, is_primary=is_primary))

    @sync_phase_handler(TimingPhase.ROUTING, True)
    def sync_routing_enter(self):
        """
        In SYNC timing mode, enter ROUTING phase.
        """
        # Transmit link states based on entangled qubits arrived during EXTERNAL phase.
        self.nb.send_link_state()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self):
        """
        In SYNC timing mode, exit INTERNAL phase.
        """
        # Clear path assignments, as these are only useful for one slot.
        self.memory.deallocate(*(qubit.addr for qubit, _ in self.memory.find(lambda q, _: q.path_id is not None)))
