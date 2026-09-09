from collections.abc import Sequence
from typing import Literal, NamedTuple, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.node import Application, Controller, Node
from mqns.network.fw.message import PathDeleteMsg, PathInsertMsg, PathInstructions, PathReachEprCountMsg
from mqns.network.fw.routing import ComputeRoutesContext, RoutingPath
from mqns.network.network import QuantumNetwork, Request, RequestInactiveEvent, RequestState


class RoutingController(ClassicCommandDispatcherMixin, Application[Controller]):
    """
    Centralized control plane that works with ``Forwarder`` subclass.
    """

    type InsertedPathHandle = "_InsertedPathHandle"
    """
    Opaque handle returned by ``path_insert()`` that allows deleting the paths.
    """

    net: QuantumNetwork
    route_ctx: ComputeRoutesContext

    def __init__(self):
        super().__init__()
        self._channel_primary = set[tuple[str, str]]()
        self._next_req_id = 0
        self._next_path_id = 0

    @override
    def install(self, node) -> None:
        self._application_install(node, Controller)
        self.net = self.node.network

        self.net.build_route()
        self.route_ctx = _ComputeRoutesContext(self)

    def prepare_path(self, req: Request) -> RoutingPath:
        """
        Ensure ``req.rp`` exists and is ready for path computation.
        Assign ``rp.req_id`` if absent.
        """
        if (rp := req.rp) is None:
            req.rp = rp = RoutingPath(req.src, req.dst, **req.rp_args)

        if rp.req_id < 0:
            rp.req_id = self._next_req_id
        self._next_req_id = max(self._next_req_id, rp.req_id + 1)

        return rp

    def _choose_ll_dir(self, a: str, b: str, /) -> Literal["R", "L"]:
        if (b, a) in self._channel_primary:
            return "L"
        self._channel_primary.add((a, b))
        return "R"

    def path_insert(self, req_id: int, insts: list[PathInstructions], *, epr_count=-1) -> InsertedPathHandle:
        """
        Send a southbound PATH_INSERT command.

        Args:
            req_id: Request identifier.
            insts: Path instructions. ``path_id`` is filled automatically.

        Returns:
            Opaque object that allows deleting the paths.
        """
        nodes = set[str]()
        for inst in insts:
            inst["path_id"] = self._next_path_id
            self._next_path_id += 1
            nodes.update(inst["route"])
        node_names = sorted(nodes)
        iph = _InsertedPathHandle(req_id, node_names, [self.net.get_node(n) for n in node_names])

        self._send_path_command(
            iph,
            PathInsertMsg(
                cmd="PATH_INSERT",
                req_id=req_id,
                epr_count=epr_count,
                paths=insts,
            ),
        )

        return iph

    def path_delete(self, iph: InsertedPathHandle) -> None:
        """
        Send a southbound PATH_DELETE command.

        Args:
            iph: Return value of ``RoutingController.path_insert()``.
        """
        self._send_path_command(
            iph,
            PathDeleteMsg(
                cmd="PATH_DELETE",
                req_id=iph.req_id,
            ),
        )

    def _send_path_command(self, iph: InsertedPathHandle, msg: PathInsertMsg | PathDeleteMsg) -> None:
        req_id, node_names, nodes = iph
        self.log_debug("%s #%s sendto %s | %s", msg["cmd"], req_id, node_names, msg)
        for node in nodes:
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


class _InsertedPathHandle(NamedTuple):
    req_id: int
    node_names: Sequence[str]
    nodes: Sequence[Node]
