#    Modified by Amar Abane for Multiverse Quantum Network Simulator
#    Date: 05/17/2025
#    Summary of changes: Adapted logic to support dynamic approaches.
#
#    This file is based on a snapshot of SimQN (https://github.com/QNLab-USTC/SimQN),
#    which is licensed under the GNU General Public License v3.0.
#
#    The original SimQN header is included below.


#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from typing import Any, Unpack, final, override

from mqns.entity.base_channel import BaseChannel, BaseChannelInitKwargs
from mqns.entity.node import Node
from mqns.simulator import Event, Time


class ClassicPacket:
    """A classical packet that is encodable as string or bytes."""

    __slots__ = ("src", "dest", "msg", "is_json")

    def __init__(self, msg: Any, *, src: Node, dest: Node):
        """
        Args:
            msg: The message content, ``str`` or ``bytes`` or JSON-serializable.
            src: Source node.
            dest: Destination node,
        """
        self.is_json, self.msg = (False, msg) if isinstance(msg, (str, bytes)) else (True, json.dumps(msg))
        self.src = src
        self.dest = dest

    def encode(self) -> bytes:
        """Encode the message to ``bytes`` if it is ``str``."""
        if isinstance(self.msg, str):
            return self.msg.encode(encoding="utf-8")
        assert isinstance(self.msg, bytes)
        return self.msg

    def get(self) -> Any:
        """Retrieve the message, decoding JSON if necessary."""
        return json.loads(self.msg) if self.is_json else self.msg

    def __len__(self) -> int:
        return len(self.msg)


class ClassicChannelInitKwargs(BaseChannelInitKwargs):
    pass


class ClassicChannel(BaseChannel[Node]):
    """
    A channel for classical communication.

    Note: while this class permits more than two nodes in a classical channel,
    most simulator modules expect exactly two nodes in a classical channel.
    """

    def __init__(self, name: str, **kwargs: Unpack[ClassicChannelInitKwargs]):
        super().__init__(name, **kwargs)

    def send(self, packet: ClassicPacket, next_hop: Node):
        """
        Send a classical packet.

        Args:
            packet: A classical packet.
            next_hop: Next hop node.

        Raises:
            LookupError: The next_hop is not connected to this channel.
        """
        drop, recv_time = self._send(packet, len(packet), next_hop)

        if drop:
            return

        event = RecvClassicPacket(self, packet, next_hop, t=recv_time)
        self.simulator.add_event(event)

    def __repr__(self) -> str:
        return "<cchannel " + self.name + ">"


@final
class RecvClassicPacket(Event):
    """The event for a Node to receive a classic packet"""

    def __init__(self, cchannel: ClassicChannel, packet: ClassicPacket, dest: Node, *, t: Time):
        super().__init__(t)
        self.cchannel = cchannel
        self.packet = packet
        self.dest = dest

    @override
    def invoke(self) -> None:
        self.dest.handle(self)
