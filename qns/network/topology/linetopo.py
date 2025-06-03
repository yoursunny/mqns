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

from qns.entity.node import QNode
from qns.entity.qchannel import QuantumChannel
from qns.network.topology.topo import Topology


class LineTopology(Topology):
    """LineTopology includes `nodes_number` Qnodes. The topology is a line pattern.
    """

    def build(self) -> tuple[list[QNode], list[QuantumChannel]]:
        nl: list[QNode] = []
        ll: list[QuantumChannel] = []
        if self.nodes_number >= 1:
            n = QNode(f"n{1}")
            nl.append(n)
        pn = n
        for i in range(self.nodes_number - 1):
            n = QNode(f"n{i+2}")
            nl.append(n)
            link = QuantumChannel(name=f"l{i+1}", **self.qchannel_args)
            ll.append(link)

            pn.add_qchannel(link)
            n.add_qchannel(link)
            pn = n

        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll
