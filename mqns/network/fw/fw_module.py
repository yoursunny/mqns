from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from mqns.entity.cchannel import ClassicCommandModule, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import Application, Node, QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import Fib, FibPath
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator
from mqns.utils import LogSelfMixin, unwrap

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder, ForwarderCounters


def fw_control_cmd_handler(cmd: str):
    """
    Method decorator for a control message handler in Forwarder.

    ``handle_message(self, msg: dict) -> Any``
    """

    def decorator(f: Callable[[Any, Any], Any]):
        def fw_control_cmd_handler_wrapper(self: "Forwarder|ForwarderModule", pkt: ClassicPacket, msg: Mapping):
            self.log_debug("RECV_%s from=%s | %s", cmd, pkt.src.name, msg)
            f(self, msg)

        return classic_cmd_handler(cmd)(fw_control_cmd_handler_wrapper)

    return decorator


def fw_signaling_cmd_handler(cmd: str):
    """
    Method decorator for a signaling message handler in Forwarder.

    ``handle_message(self, msg: dict, fp: FibPath) -> Any``
    """

    def decorator(f: Callable[[Any, Any, FibPath], Any]):
        def fw_signaling_cmd_handler_wrapper(self: "Forwarder|ForwarderModule", pkt: ClassicPacket, msg: Mapping):
            path_id: int = msg["path_id"]
            src_name = pkt.src.name
            dest_name = pkt.dest.name
            try:
                fp = self.fib.get_path(path_id)
            except LookupError:
                self.log_debug("DROP_%s from=%s reason=no-fib-path | %s", cmd, src_name, msg)
                return

            if pkt.dest is self.node:
                self.log_debug("RECV_%s from=%s | %s", cmd, src_name, msg)
                f(self, msg, fp)
            else:
                next_hop, via_str = _find_next_hop(self.network, fp, dest_name)
                self.log_debug("FORW_%s from=%s %sto=%s path=%s | %s", cmd, src_name, via_str, dest_name, path_id, msg)
                self.node.send_cpacket(next_hop, pkt)

        return classic_cmd_handler(cmd)(fw_signaling_cmd_handler_wrapper)

    return decorator


def _find_next_hop(net: QuantumNetwork, fp: FibPath, dest_name: str) -> tuple[Node, str]:
    dest_idx = fp.route.index(dest_name)
    nh_idx = fp.own_idx + 1 if dest_idx > fp.own_idx else fp.own_idx - 1
    nh_name = fp.route[nh_idx]
    return net.get_node(nh_name), "" if nh_idx == dest_idx else f"via={nh_name} "


class ForwarderModule(LogSelfMixin, ClassicCommandModule):
    """
    Base class of a module within the forwarder.
    """

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory
    fib: Fib
    fw_cnt: "ForwarderCounters"

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory
        self.fib = fw.fib
        self.fw_cnt = fw.cnt

    def __repr__(self):
        return Application.__repr__(self)

    def send_ctrl(self, msg: Mapping) -> None:
        """
        Send a control message to the network controller.
        """
        ctrl = unwrap(self.network.controller)
        self.log_debug("SEND_%s to=%s | %s", msg["cmd"], ctrl.name, msg)
        self.node.send_cpacket(ctrl, ClassicPacket(msg, src=self.node, dest=ctrl))

    def send_msg(self, dest: Node, msg: Mapping, fp: FibPath) -> None:
        """
        Send a signaling message along the path specified in FIB path entry.
        """
        dest_name = dest.name
        next_hop, via_str = _find_next_hop(self.network, fp, dest_name)
        self.log_debug("SEND_%s %sto=%s path=%s | %s", msg["cmd"], via_str, dest_name, fp.path_id, msg)

        pkt = ClassicPacket(msg, src=self.node, dest=dest)
        self.node.send_cpacket(next_hop, pkt)
