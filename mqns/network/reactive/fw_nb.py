from mqns.entity.memory import MemoryQubit, PathDirection
from mqns.entity.node import QNode
from mqns.network.fw import FibPath, FibRequest, ForwarderNorthbound, fw_control_cmd_handler
from mqns.network.fw.message import PathInsertMsg, PathInstructions, validate_path_instructions
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg
from mqns.simulator import Time
from mqns.utils import unwrap


class ReactiveForwarderNorthbound(ForwarderNorthbound):
    """
    Northbound interface to communicate with ``ReactiveRoutingController``.
    """

    def __init__(self):
        super().__init__()

        self._link_states: list[LinkStateEntry] = []
        """
        LinkState accumulated during EXTERNAL phase and sent at start of ROUTING phase.
        """

    def append_link_state(self, neighbor: QNode, mq: MemoryQubit):
        self._link_states.append(
            {
                "node": self.node.name,
                "neighbor": neighbor.name,
                "qubit": unwrap(mq.key),
            }
        )

    def send_link_state(self):
        """
        Send link state message to controller. Assumes direct connection to controller.
        """
        if len(self._link_states) == 0:
            self.log_debug("no link_state to send")
            return
        else:
            self.log_debug("send link_state for %s entries", len(self._link_states))

        msg: LinkStateMsg = {
            "cmd": "LS",
            "ls": self._link_states,
        }
        self.send_ctrl(msg)

        self._link_states.clear()

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
            for dir, _ in self.iter_adjacency(fp):
                self._install_path_adj(inst, fp, dir)

    def _path_convert(self, inst: PathInstructions) -> FibPath:
        validate_path_instructions(inst, reactive=True)
        route = inst["route"]

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

    def _install_path_adj(self, inst: PathInstructions, fp: FibPath, dir: PathDirection) -> None:
        assert "reactive_qubits" in inst
        idx = fp.own_idx + (-1 if dir == PathDirection.L else 0)
        key = inst["reactive_qubits"][idx]

        try:
            qubit, _ = next(self.memory.find(lambda q, _: q.key == key))
        except StopIteration:
            raise ValueError(f"reactive_qubits[{idx}] refers to non-existent qubit {key}")

        qubit.path_id, qubit.path_direction = fp.path_id, dir
        addrs = [qubit.addr]
        self.fw.log_debug("allocating %s qubits: %s", dir, addrs)
