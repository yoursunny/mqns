import functools
from collections.abc import Callable, Mapping
from typing import Any

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.node import Node, QNode
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.network import QuantumNetwork
from mqns.utils import log


def fw_control_cmd_handler(cmd: str):
    """
    Method decorator for a control message handler in Forwarder.

    ``handle_message(self, msg: dict) -> Any``
    """

    def decorator(f: Callable[[Any, Any], Any]):
        @functools.wraps(f)
        def wrapper(self: "ForwarderClassicMixin", pkt: ClassicPacket, msg: dict):
            log.debug("%s: received control message from %s | %s", self, pkt.src.name, msg)
            f(self, msg)
            return True

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


def fw_signaling_cmd_handler(cmd: str):
    """
    Method decorator for a signaling message handler in Forwarder.

    ``handle_message(self, msg: dict, fib_entry: FibEntry) -> Any``
    """

    def decorator(f: Callable[[Any, Any, FibEntry], Any]):
        @functools.wraps(f)
        def wrapper(self: "ForwarderClassicMixin", pkt: ClassicPacket, msg: dict):
            path_id: int = msg["path_id"]
            try:
                fib_entry = self.fib.get(path_id)
            except LookupError:
                log.debug("%s: dropping signaling message from %s, reason=no-fib-entry | %s", self, pkt.src.name, msg)
                return True

            if pkt.dest != self.node:
                self.send_msg(pkt.dest, msg, fib_entry, forward_from=pkt.src)
                return True

            log.debug("%s: received signaling message from %s | %s", self, pkt.src.name, msg)
            f(self, msg, fib_entry)
            return True

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


class ForwarderClassicMixin(ClassicCommandDispatcherMixin):
    """
    Part of ``Forwarder`` logic related to classical message handling.

    * Dispatch classical control and signaling messages.
    * Forward classical signaling messages according to path_id lookup in FIB.
    """

    __slots__ = ()
    node: QNode
    network: QuantumNetwork
    fib: Fib

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
