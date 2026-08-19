#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
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

from typing import TYPE_CHECKING, override

from mqns.entity.node.app import Application
from mqns.entity.node.node import Node

if TYPE_CHECKING:
    from mqns.entity.memory import QuantumMemory
    from mqns.entity.operator import QuantumOperator
    from mqns.entity.qchannel import QuantumChannel


class QNode(Node):
    """QNode is a quantum node in the quantum network. Inherits Node and add quantum elements."""

    def __init__(self, name: str, *, apps: list[Application] | None = None):
        """
        Args:
            name: node name
            apps: applications on the node.
        """
        super().__init__(name, apps=apps)
        self.qchannels: list[QuantumChannel] = []
        """Quantum channels connected to this node."""
        self._qchannel_by_dst: dict[Node, QuantumChannel] = {}
        self._memory: QuantumMemory | None = None
        self.operators: list[QuantumOperator] = []

    @override
    def _node_install(self) -> None:
        self._install_channels(self.qchannels, self._qchannel_by_dst)

        if self._memory:
            self._memory.install(self.simulator)
        for operator in self.operators:
            operator.install(self.simulator)

    @property
    def memory(self) -> "QuantumMemory":
        """
        Retrieve associated QuantumMemory.

        Raises:
            LookupError: Memory does not exist.
        """
        if self._memory is None:
            raise LookupError(f"node {self} does not have memory")
        return self._memory

    @memory.setter
    def memory(self, value: "QuantumMemory"):
        """
        Assign QuantumMemory to this node.
        This setter is available prior to calling .install().
        """
        self.ensure_not_installed()
        value.node = self
        self._memory = value

    def add_operator(self, operator: "QuantumOperator"):
        """Add a quantum operator in this node

        Args:
            operator: the quantum operator

        This function is available prior to calling .install().
        """
        self.ensure_not_installed()
        operator.set_own(self)
        self.operators.append(operator)

    def add_qchannel(self, qchannel: "QuantumChannel"):
        """
        Add a quantum channel in this QNode.
        This function is available prior to calling .install().
        """
        self._add_channel(qchannel, self.qchannels)

    def get_qchannel(self, dst: "QNode") -> "QuantumChannel":
        """
        Retrieve the quantum channel that connects to ``dst``.

        Raises:
            LookupError: channel does not exist
        """
        return self._get_channel(dst, self._qchannel_by_dst)

    def __repr__(self) -> str:
        return f"<qnode {self.name}>"
