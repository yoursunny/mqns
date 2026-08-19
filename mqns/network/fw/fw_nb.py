from abc import abstractmethod
from typing import TYPE_CHECKING

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.fib import FibPath, FibRequest
from mqns.network.fw.fw_module import ForwarderModule, fw_control_cmd_handler
from mqns.network.fw.message import PathDeleteMsg, PathInsertMsg, PathInstructions, PathReachEprCountMsg
from mqns.network.fw.mux import MuxScheme
from mqns.simulator import Time

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
        self._fib_erase_delay = self.memory.t_cohere * 4

    @fw_control_cmd_handler("PATH_INSERT")
    def handle_path_insert(self, msg: PathInsertMsg) -> None:
        """Process a PATH_INSERT control command."""
        paths = [(inst, self._path_convert(inst)) for inst in msg["paths"] if self.node.name in inst["route"]]

        fr = FibRequest(msg["req_id"], [entry for _, entry in paths], epr_count=msg.get("epr_count", -1))
        self.fib.insert_req(fr)

        # Identify left/right channels, allocate qubits and process LinkLayer changes.
        for inst, entry in paths:
            if ch := self._find_adj(entry, -1):
                self.mux.install_path_adj(inst, entry, PathDirection.L, ch)
                self.install_path_adj(entry, PathDirection.L, ch)
            if ch := self._find_adj(entry, +1):
                self.mux.install_path_adj(inst, entry, PathDirection.R, ch)
                self.install_path_adj(entry, PathDirection.R, ch)

    def _path_convert(self, inst: PathInstructions) -> FibPath:
        route = inst["route"]
        self.mux.validate_path_instructions(inst)

        # Insert FIB entry.
        if "swap_cutoff" in inst:
            swap_cutoff = [None if t < 0 else self.simulator.time(slot=t) for t in inst["swap_cutoff"]]
        else:
            swap_cutoff: list[Time | None] = [None] * (2 * (len(route) - 2))
        return FibPath(
            path_id=inst["path_id"],
            route=route,
            own_idx=route.index(self.node.name),
            swap=inst["swap"],
            swap_cutoff=swap_cutoff,
            purif=inst["purif"],
        )

    @abstractmethod
    def install_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Process LinkLayer changes for an adjacency after a path has been installed.
        """

    @fw_control_cmd_handler("PATH_DELETE")
    def handle_path_delete(self, msg: PathDeleteMsg) -> None:
        """Process a PATH_DELETE control command."""
        fr = self.fib.delete_req(msg["req_id"], delay=self._fib_erase_delay)

        # Identify left/right channels, deallocate qubits and process LinkLayer changes.
        for fp in fr.paths:
            if ch := self._find_adj(fp, -1):
                self.mux.uninstall_path_adj(fp, PathDirection.L, ch)
                self.uninstall_path_adj(fp, PathDirection.L, ch)
            if ch := self._find_adj(fp, +1):
                self.mux.uninstall_path_adj(fp, PathDirection.R, ch)
                self.uninstall_path_adj(fp, PathDirection.R, ch)

    @abstractmethod
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        """
        Process LinkLayer changes for an adjacency after a path has been installed.
        """

    def _find_adj(self, fp: FibPath, route_offset: int) -> QuantumChannel | None:
        neigh_idx = fp.own_idx + route_offset
        if neigh_idx in (-1, len(fp.route)):  # no left/right neighbor if own node is the left/right end node
            return None
        neigh = self.network.get_node(fp.route[neigh_idx])
        return self.node.get_qchannel(neigh)

    def send_reach_epr_count(self, fr: FibRequest) -> None:
        msg = PathReachEprCountMsg(
            cmd="PATH_REACH_EPR_COUNT",
            req_id=fr.req_id,
        )
        self.send_ctrl(msg)
