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

from typing import Final, Unpack, override

from mqns.entity.memory import MemoryQubit, QubitState
from mqns.models.epr import Entanglement
from mqns.network.fw import FibPath, Forwarder, ForwarderInitKwargs, MuxScheme
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.network.protocol.event import PathActivateEvent, QubitEntangledEvent
from mqns.network.reactive.fw_nb import ReactiveForwarderNorthbound
from mqns.utils import unwrap, unwrap_cast


class ReactiveForwarder(Forwarder):
    """
    ReactiveForwarder is the forwarder of QNodes. It continuously generates EPRs on all links,
    then sends EPR link state to the controller to receive routing instructions.
    Works only in Synchronous timing mode.
    It implements the forwarding phase (i.e., entanglement generation and swapping) while the centralized
    routing is done at the controller.
    """

    nb: Final[ReactiveForwarderNorthbound]
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
            self.simulator.sched(PathActivateEvent(self.node, ch, None, t=self.simulator.tc, is_primary=is_primary))

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
        # Clear FIB and path assignments, as these are only useful for one slot.
        self.fib.clear()
        self.memory.deallocate(*(qubit.addr for qubit, _ in self.memory.find(lambda q, _: q.path_id is not None)))

    @override
    def qubit_is_entangled_in_external(self, event: QubitEntangledEvent) -> None:
        """
        Handle a qubit entering ENTANGLED state when in EXTERNAL phase of SYNC timing mode.
        """
        super().qubit_is_entangled_in_external(event)
        self.nb.append_link_state(event.neighbor, event.qubit)

    @override
    def qubit_is_entangled_next(self, mq: MemoryQubit, epr: Entanglement) -> FibPath | None:
        if mq.path_id is None:  # qubit not used in PATH_INSERT
            return
        mq.epr_path_ids = [mq.path_id]

        mq.state = QubitState.PURIF
        return self.fib.get_path(unwrap_cast(mq.path_id))

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        _ = epr0
        try:
            mq1, _ = next(
                self.memory.find(
                    lambda q, _: (
                        MuxScheme.qubits_swappable(mq0, q)  # basic condition met
                        and q.path_id == mq0.path_id  # allocated to the same path_id
                        and q.path_direction is not mq0.path_direction  # in the opposite path direction
                    ),
                    has=self.epr_type,
                )
            )
            return mq1, unwrap(fp)
        except StopIteration:
            return None
