from typing import Literal, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.node import Application, Controller
from mqns.network.fw.message import (
    PathDeleteMsg,
    PathInsertMsg,
    PathInstructions,
    PathReachEprCountMsg,
    validate_path_instructions,
)
from mqns.network.fw.routing import ComputeRoutesContext, MultiplexingVectorInput, RoutingPath
from mqns.network.network import QuantumNetwork, RequestInactiveEvent, RequestState


class RoutingController(ClassicCommandDispatcherMixin, Application[Controller]):
    """
    Centralized control plane that works with ``Forwarder`` subclass.
    """

    net: QuantumNetwork
    route_ctx: ComputeRoutesContext

    def __init__(self, *, mv_auto: MultiplexingVectorInput = "none"):
        """
        Args:
            mv_auto: How to interpret ``RoutingPath(bufferspace_mv="auto")``.
                     This should be set to ``max`` if forwarders use ``MuxSchemeBufferSpace``, otherwise ``none``.
        """
        super().__init__()
        self.mv_auto: MultiplexingVectorInput = mv_auto
        self._channel_primary = set[tuple[str, str]]()

    @override
    def install(self, node) -> None:
        self._application_install(node, Controller)
        self.net = self.node.network
        self._next_req_id = 0
        self._next_path_id = 0

        self.net.build_route()
        self.route_ctx = _ComputeRoutesContext(self)

    def prepare_path(self, rp: RoutingPath) -> None:
        """
        Ensure ``rp`` is ready for path computation.

        * Assign ``rp.req_id`` and ``rp.path_id`` if absent.
        * Replace ``rp.bufferspace_mv="auto"`` with a concrete value.
        """
        if rp.req_id < 0:
            rp.req_id = self._next_req_id
        self._next_req_id = max(self._next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self._next_path_id

        if rp.bufferspace_mv == "auto":
            rp.bufferspace_mv = self.mv_auto

    def _choose_ll_dir(self, a: str, b: str, /) -> Literal["R", "L"]:
        if (b, a) in self._channel_primary:
            return "L"
        self._channel_primary.add((a, b))
        return "R"

    def install_path(self, rp: RoutingPath, *, recompute: bool, epr_count=-1) -> None:
        """
        Compute routing path(s) and send PATH_INSERT commands to nodes.

        Args:
            recompute: If True, always make ``rp`` re-compute path instructions.
                       If False, allow reusing previously computed paths cached in ``rp``.
            epr_count: Desired EPR count to include in PATH_INSERT messages.
        """
        self.prepare_path(rp)

        insts: list[PathInstructions] = []
        nodes = set[str]()
        for inst in rp.list_paths(self.route_ctx, recompute=recompute):
            self._next_path_id = max(self._next_path_id, inst["path_id"] + 1)
            validate_path_instructions(inst, bufferspace=None, reactive=None)
            insts.append(inst)
            nodes.update(inst["route"])

        self._send_path_command(
            nodes,
            PathInsertMsg(
                cmd="PATH_INSERT",
                req_id=rp.req_id,
                epr_count=epr_count,
                paths=insts,
            ),
        )

    def uninstall_path(self, rp: RoutingPath) -> None:
        """
        Compute routing path(s) and send PATH_DELETE commands to nodes.
        """
        assert rp.req_id >= 0
        assert rp.path_id >= 0

        nodes = set[str]()
        for inst in rp.list_paths(self.route_ctx, recompute=False):
            nodes.update(inst["route"])

        self._send_path_command(
            nodes,
            PathDeleteMsg(
                cmd="PATH_DELETE",
                req_id=rp.req_id,
            ),
        )

    def _send_path_command(self, nodes: set[str], msg: PathInsertMsg | PathDeleteMsg) -> None:
        node_list = sorted(nodes)  # ensure deterministic order
        self.log_debug("%s #%s sendto %s | %s", msg["cmd"], msg["req_id"], node_list, msg)
        for node_name in node_list:
            node = self.net.get_node(node_name)
            self.node.send_cpacket(node, ClassicPacket(msg, src=self.node, dest=node))

    @classic_cmd_handler("PATH_REACH_EPR_COUNT")
    def handle_reach_epr_count(self, pkt: ClassicPacket, msg: PathReachEprCountMsg) -> None:
        req_id = msg["req_id"]
        end_node = pkt.src.name

        req = next((req for req in self.net.requests if req.req_id == req_id), None)
        if req is None:
            self.log_debug("reach_epr_count req=%s end_node=%s outcome=req-not-found", req_id, end_node)
            return

        if req.epr_count_await is None:
            self.log_debug("reach_epr_count req=%s end_node=%s outcome=epr-count-unrestricted", req_id, end_node)
            return

        try:
            req.epr_count_await.remove(end_node)
        except KeyError:
            self.log_debug("reach_epr_count req=%s end_node=%s outcome=node-not-pending", req_id, end_node)
            return

        if req.epr_count_await:
            self.log_debug(
                "reach_epr_count req=%s end_node=%s outcome=wait-for-other-end await=%s", req_id, end_node, req.epr_count_await
            )
            return

        self.log_debug("reach_epr_count req=%s end_node=%s outcome=deactivate-request", req_id, end_node)
        req.state = RequestState.EPR_COUNT_REACHED
        self.simulator.sched(event := RequestInactiveEvent(self.node, req, t=self.simulator.tc))
        req.inactive_event.set(event)


class _ComputeRoutesContext:
    def __init__(self, ctrl: RoutingController):
        self.time_accuracy = ctrl.net.simulator.accuracy
        self.get_qchannel = ctrl.net.get_qchannel
        self.query_route = ctrl.net.query_route
        self.choose_ll_dir = ctrl._choose_ll_dir
