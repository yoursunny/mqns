from typing import override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.node import Application, Controller
from mqns.network.fw.message import (
    PathDeleteMsg,
    PathInsertMsg,
    PathInstructions,
    PathReachEprCountMsg,
    validate_path_instructions,
)
from mqns.network.fw.routing import MultiplexingVectorInput, RoutingPath
from mqns.network.network import RequestInactiveEvent


class RoutingController(ClassicCommandDispatcherMixin, Application[Controller]):
    """
    Centralized control plane that works with ``Forwarder`` subclass.
    """

    def __init__(self, *, mv_auto: MultiplexingVectorInput):
        """
        Args:
            mv_auto: How to interpret ``RoutingPath(bufferspace_mv="auto")``.
                     This should be set to ``max`` if forwarders use ``MuxSchemeBufferSpace``, otherwise ``none``.
        """
        super().__init__()
        self.mv_auto: MultiplexingVectorInput = mv_auto

    @override
    def install(self, node) -> None:
        self._application_install(node, Controller)
        self.net = self.node.network
        self._next_req_id = 0
        self._next_path_id = 0

        self.net.build_route()

    def install_path(self, rp: RoutingPath, *, epr_count=-1) -> None:
        """
        Compute routing path(s) and send PATH_INSERT commands to nodes.
        """
        if rp.req_id < 0:
            rp.req_id = self._next_req_id
        self._next_req_id = max(self._next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self._next_path_id

        if rp.bufferspace_mv == "auto":
            rp.bufferspace_mv = self.mv_auto

        insts: list[PathInstructions] = []
        nodes = set[str]()
        for inst in rp.list_paths(self.net, recompute=True):
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
        for inst in rp.list_paths(self.net):
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
        self.simulator.sched(event := RequestInactiveEvent(self.node, req, t=self.simulator.tc))
        req.inactive_event.set(event)
