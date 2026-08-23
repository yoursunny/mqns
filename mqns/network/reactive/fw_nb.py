from mqns.network.fw import (
    FibPath,
    FibRequest,
    Forwarder,
    ForwarderNorthbound,
    MuxScheme,
    fw_control_cmd_handler,
)
from mqns.network.fw.message import PathInsertMsg, PathInstructions
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.simulator import Time


class ReactiveForwarderNorthbound(ForwarderNorthbound):
    """
    Northbound interface to communicate with ``ReactiveRoutingController``.
    """

    mux: MuxScheme

    def install(self, fw: Forwarder):
        super().install(fw)
        self.mux = fw.mux

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

    @fw_control_cmd_handler("PATH_INSERT")
    def handle_path_insert(self, msg: PathInsertMsg) -> None:
        """Process a PATH_INSERT control command."""
        if not self.node.timing.is_routing():
            # The likely cause is setting t_rtg too short in TimingModeSync.
            self.log_warning("PATH_INSERT(req_id=%s) ignored reason=not-routing-phase", msg["req_id"])
            return

        paths = [(inst, self._path_convert(inst)) for inst in msg["paths"] if self.node.name in inst["route"]]

        fr = FibRequest(msg["req_id"], [fp for _, fp in paths], epr_count=msg.get("epr_count", -1))
        self.fib.insert_req(fr)

        # Identify left/right channels, allocate qubits and process LinkLayer changes.
        for inst, fp in paths:
            for dir, ch in self.iter_adjacency(fp):
                self.mux.install_path_adj(inst, fp, dir, ch)

    def _path_convert(self, inst: PathInstructions) -> FibPath:
        route = inst["route"]
        self.mux.validate_path_instructions(inst)

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
