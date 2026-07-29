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
    def handle_install_path(self, msg: InstallPathMsg) -> None:
        """Process an INSTALL_PATH control command."""
        path_id = msg["path_id"]
        instructions = msg["instructions"]
        self.mux.validate_path_instructions(instructions)

        # Insert FIB entry.
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

        # Identify left/right channels, allocate qubits and process LinkLayer changes.
        if ch := self._find_adj(fib_entry, -1):
            self.mux.install_path_adj(instructions, fib_entry, PathDirection.L, ch)
            self.install_path_adj(fib_entry, PathDirection.L, ch)
        if ch := self._find_adj(fib_entry, +1):
            self.mux.install_path_adj(instructions, fib_entry, PathDirection.R, ch)
            self.install_path_adj(fib_entry, PathDirection.R, ch)

    @abstractmethod
    def install_path_adj(self, fib_entry: FibEntry, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Process LinkLayer changes for an adjacency after a path has been installed.
        """

    @fw_control_cmd_handler("UNINSTALL_PATH")
    def handle_uninstall_path(self, msg: UninstallPathMsg) -> None:
        """Process an UNINSTALL_PATH control command."""
        path_id = msg["path_id"]

        # Retrieve and erase FIB entry.
        fib_entry = self.fib.get(path_id)
        fib_entry.active_until = self.simulator.tc
        self.simulator.add_event(func_to_event(fib_entry.active_until + self._fib_erase_delay, self.fib.erase, path_id))

        # Identify left/right channels, deallocate qubits and process LinkLayer changes.
        if ch := self._find_adj(fib_entry, -1):
            self.mux.uninstall_path_adj(fib_entry, PathDirection.L, ch)
            self.uninstall_path_adj(fib_entry, PathDirection.L, ch)
        if ch := self._find_adj(fib_entry, +1):
            self.mux.uninstall_path_adj(fib_entry, PathDirection.R, ch)
            self.uninstall_path_adj(fib_entry, PathDirection.R, ch)

    @abstractmethod
    def uninstall_path_adj(self, fib_entry: FibEntry, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Process LinkLayer changes for an adjacency after a path has been installed.
        """

    def _find_adj(self, fib_entry: FibEntry, route_offset: int) -> QuantumChannel | None:
        neigh_idx = fib_entry.own_idx + route_offset
        if neigh_idx in (-1, len(fib_entry.route)):  # no left/right neighbor if own node is the left/right end node
            return None
        neigh = self.network.get_node(fib_entry.route[neigh_idx])
        return self.node.get_qchannel(neigh)
