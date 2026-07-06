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

from mqns.entity.cchannel import RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.network.route import RouteAlgorithm
from mqns.simulator import event_handler


class ClassicPacketForwardApp(Application[Node]):
    """This application will generate routing table for classic networks
    and allow nodes to forward classic packets to the destination.
    """

    def __init__(self, route: RouteAlgorithm):
        """
        Args:
            route: routing algorithm
        """
        super().__init__()
        self.route = route

    @event_handler
    def handleClassicPacket(self, event: RecvClassicPacket):
        packet = event.packet

        dst = packet.dest
        if dst == self.node:
            # The destination is this node, return to let later application to handle this packet
            return False

        # If destination is not this node, forward this packet
        routes = self.route.query(self.node, dst)
        if not routes:
            # no routing result or error format, drop this packet
            return True
        next_hop = routes[0].next_hop
        try:
            self.node.send_cpacket(next_hop, packet)
        except LookupError:
            # not found the classic channel, drop the packet
            return True
        return True
