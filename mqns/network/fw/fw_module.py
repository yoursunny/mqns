import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from mqns.entity.cchannel import ClassicCommandModule, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import QuantumMemory
from mqns.entity.node import Application, Node, QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator
from mqns.utils import LogSelfMixin, log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder, ForwarderCounters


def fw_control_cmd_handler(cmd: str):
    """
    Method decorator for a control message handler in Forwarder.

    ``handle_message(self, msg: dict) -> Any``
    """

    def decorator(f: Callable[[Any, Any], Any]):
        @functools.wraps(f)
        def wrapper(self: "Forwarder|ForwarderModule", pkt: ClassicPacket, msg: dict):
            self.log_debug("received control message from %s | %s", pkt.src.name, msg)
            f(self, msg)

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


def fw_signaling_cmd_handler(cmd: str):
    """
    Method decorator for a signaling message handler in Forwarder.

    ``handle_message(self, msg: dict, fib_entry: FibEntry) -> Any``
    """

    def decorator(f: Callable[[Any, Any, FibEntry], Any]):
        @functools.wraps(f)
        def wrapper(self: "ForwarderModule", pkt: ClassicPacket, msg: dict):
            path_id: int = msg["path_id"]
            try:
                fib_entry = self.fib.get(path_id)
            except LookupError:
                self.log_debug("dropping signaling message from %s, reason=no-fib-entry | %s", pkt.src.name, msg)
                return

            if pkt.dest != self.node:
                self.send_msg(pkt.dest, msg, fib_entry, forward_from=pkt.src)
                return

            self.log_debug("received signaling message from %s | %s", pkt.src.name, msg)
            f(self, msg, fib_entry)

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


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

    def send_ctrl(self, msg: Mapping):
        ctrl = self.network.get_controller()
        log.debug("%s: sending control message to controller | %s", self, msg)
        self.node.send_cpacket(ctrl, ClassicPacket(msg, src=self.node, dest=ctrl))

    def send_msg(self, dest: Node, msg: Mapping, fib_entry: FibEntry, *, forward_from: Node | None = None):
        """
        Send/forward a signaling message along the path specified in FIB entry.
        """
        dest_idx = fib_entry.route.index(dest.name)
        nh_idx = fib_entry.own_idx + 1 if dest_idx > fib_entry.own_idx else fib_entry.own_idx - 1
        next_hop = self.network.get_node(fib_entry.route[nh_idx])

        pkt = ClassicPacket(msg, src=forward_from or self.node, dest=dest)
        log.debug(
            "%s: %s signaling message from %s to %s%s | %s",
            self,
            "forwarding" if forward_from else "sending",
            pkt.src.name,
            pkt.dest.name,
            "" if nh_idx == dest_idx else f" via {next_hop.name}",
            msg,
        )
        self.node.send_cpacket(next_hop, pkt)
