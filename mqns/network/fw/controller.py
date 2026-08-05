from typing import override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.node import Application, Controller
from mqns.network.fw.message import PathDeleteMsg, PathInsertMsg, PathInstructions
from mqns.network.fw.routing import MultiplexingVectorInput, RoutingPath


class RoutingController(Application[Controller]):
    """
    Centralized control plane that works with ``Forwarder`` subclass.
    """

    def __init__(self, *, mv_auto: MultiplexingVectorInput):
        """
        Args:
            mv_auto: How to interpret ``RoutingPath(m_v="auto")``.
                     This should be set to ``max`` if forwarders use ``MuxSchemeBufferSpace``, otherwise ``none``.
        """
        super().__init__()
        self.mv_auto = mv_auto

    @override
    def install(self, node):
        self._application_install(node, Controller)
        self.net = self.node.network
        self._next_req_id = 0
        self._next_path_id = 0

        self.net.build_route()

    def install_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and send PATH_INSERT commands to nodes.
        """
        if rp.req_id < 0:
            rp.req_id = self._next_req_id
        self._next_req_id = max(self._next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self._next_path_id

        if rp.m_v == "auto":
            rp.m_v = self.mv_auto

        insts: list[PathInstructions] = []
        nodes = set[str]()
        for path_id, inst in enumerate(rp.compute_paths(self.net), start=rp.path_id):
            self._next_path_id = max(self._next_path_id, path_id + 1)
            inst["path_id"] = path_id
            insts.append(inst)
            nodes.update(inst["route"])

        self._send_path_command(
            nodes,
            PathInsertMsg(
                cmd="PATH_INSERT",
                req_id=rp.req_id,
                paths=insts,
            ),
        )

    def uninstall_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and send PATH_DELETE commands to nodes.
        """
        assert rp.req_id >= 0
        assert rp.path_id >= 0

        nodes = set[str]()
        for inst in rp.compute_paths(self.net):
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
