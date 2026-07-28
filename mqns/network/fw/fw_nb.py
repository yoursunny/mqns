from abc import abstractmethod
from typing import TYPE_CHECKING

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.fw_module import ForwarderModule, fw_control_cmd_handler
from mqns.network.fw.message import InstallPathMsg, UninstallPathMsg
from mqns.network.fw.mux import MuxScheme
from mqns.simulator import Time, func_to_event

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class ForwarderNorthbound(ForwarderModule):
    """
    Northbound interface of the forwarder to communicate with the controller.
    """

    mux: MuxScheme

    def install(self, fw: "Forwarder"):
        super().install(fw)
        self.mux = fw.mux
        self._fib_erase_delay = self.simulator.time(time_slot=4 * self.memory.t_cohere.time_slot)

    @fw_control_cmd_handler("INSTALL_PATH")
    def handle_install_path(self, msg: InstallPathMsg):
        """Process an INSTALL_PATH control command."""
        path_id = msg["path_id"]
        instructions = msg["instructions"]
        self.mux.validate_path_instructions(instructions)

        # populate FIB
        route = instructions["route"]
        if "swap_cutoff" in instructions:
            swap_cutoff = [None if t < 0 else self.simulator.time(time_slot=t) for t in instructions["swap_cutoff"]]
        else:
            swap_cutoff: list[Time | None] = [None] * (2 * (len(route) - 2))
        fib_entry = FibEntry(
            req_id=instructions["req_id"],
            path_id=path_id,
            route=route,
            own_idx=route.index(self.node.name),
            swap=instructions["swap"],
            swap_cutoff=swap_cutoff,
            purif=instructions["purif"],
        )
        self.fib.insert_or_replace(fib_entry)

        # identify left/right neighbors
        # associate path with qchannel and allocate qubits
        if ch_l := self._find_adj(fib_entry, -1):
            self.mux.install_path_adj(instructions, fib_entry, PathDirection.L, ch_l)
        if ch_r := self._find_adj(fib_entry, +1):
            self.mux.install_path_adj(instructions, fib_entry, PathDirection.R, ch_r)

        # call subclass specialization
        self.handle_path_change(
            path_id=path_id,
            uninstall=False,
            fib_entry=fib_entry,
            ch_l=ch_l,
            ch_r=ch_r,
        )

    @fw_control_cmd_handler("UNINSTALL_PATH")
    def handle_uninstall_path(self, msg: UninstallPathMsg):
        """Process an UNINSTALL_PATH control command."""
        path_id = msg["path_id"]

        # retrieve and erase FIB entry
        fib_entry = self.fib.get(path_id)
        fib_entry.active_until = self.simulator.tc
        self.simulator.add_event(func_to_event(fib_entry.active_until + self._fib_erase_delay, self.fib.erase, path_id))

        # identify left/right neighbors
        # disassociate path with qchannel and deallocate qubits
        if ch_l := self._find_adj(fib_entry, -1):
            self.mux.uninstall_path_adj(fib_entry, PathDirection.L, ch_l)
        if ch_r := self._find_adj(fib_entry, +1):
            self.mux.uninstall_path_adj(fib_entry, PathDirection.R, ch_r)

        # call subclass specialization
        self.handle_path_change(
            path_id=path_id,
            uninstall=True,
            fib_entry=fib_entry,
            ch_l=ch_l,
            ch_r=ch_r,
        )

    def _find_adj(self, fib_entry: FibEntry, route_offset: int) -> QuantumChannel | None:
        neigh_idx = fib_entry.own_idx + route_offset
        if neigh_idx in (-1, len(fib_entry.route)):  # no left/right neighbor if own node is the left/right end node
            return None
        neigh = self.network.get_node(fib_entry.route[neigh_idx])
        return self.node.get_qchannel(neigh)

    @abstractmethod
    def handle_path_change(
        self,
        *,
        path_id: int,
        uninstall: bool,
        fib_entry: FibEntry,
        ch_l: QuantumChannel | None,
        ch_r: QuantumChannel | None,
    ):
        """
        Process LinkLayer changes after a path has been installed or uninstalled.

        Args:
            path_id: Path identifier.
            uninstall: Whether this is an uninstall command.
            fib_entry: FIB entry.
            ch_l: Quantum channel toward left, if exists.
            ch_r: Quantum channel toward right, if exists.
        """
