from typing import override

from mqns.entity.cchannel import ClassicPacket
from mqns.entity.node import Application, Controller
from mqns.network.fw.message import InstallPathMsg, PathInstructions, UninstallPathMsg
from mqns.network.fw.routing import MultiplexingVectorInput, RoutingPath
from mqns.utils import log


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
        Compute routing path(s) and send install commands to nodes.
        """
        if rp.req_id < 0:
            rp.req_id = self._next_req_id
        self._next_req_id = max(self._next_req_id, rp.req_id + 1)

        if rp.path_id < 0:
            rp.path_id = self._next_path_id

        if rp.m_v == "auto":
            rp.m_v = self.mv_auto

        for path_id, instructions in enumerate(rp.compute_paths(self.net), start=rp.path_id):
            self._next_path_id = max(self._next_path_id, path_id + 1)
            self._send_instructions(
                InstallPathMsg(cmd="INSTALL_PATH", path_id=path_id, instructions=instructions),
                instructions,
            )

    def uninstall_path(self, rp: RoutingPath):
        """
        Compute routing path(s) and send uninstall commands to nodes.
        """
        assert rp.req_id >= 0
        assert rp.path_id >= 0

        for path_id_add, instructions in enumerate(rp.compute_paths(self.net)):
            self._send_instructions(UninstallPathMsg(cmd="UNINSTALL_PATH", path_id=rp.path_id + path_id_add), instructions)

    def _send_instructions(self, msg: InstallPathMsg | UninstallPathMsg, instructions: PathInstructions):
        for node_name in instructions["route"]:
            qnode = self.net.get_node(node_name)
            self.node.send_cpacket(qnode, ClassicPacket(msg, src=self.node, dest=qnode))
            log.debug(f"{self}: {msg['cmd']} #{msg['path_id']} at {qnode.name}: {instructions}")
