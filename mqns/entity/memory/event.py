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


from typing import TYPE_CHECKING, final, override

from mqns.entity.memory.memory_qubit import MemoryQubit
from mqns.models.core import QuantumModel
from mqns.simulator import Event, Time

if TYPE_CHECKING:
    from mqns.entity.memory.memory import QuantumMemory


@final
class MemoryDecohereEvent(Event):
    """Event sent by QuantumMemory to inform LinkLayer about a decohered qubit."""

    def __init__(
        self,
        memory: "QuantumMemory",
        qubit: MemoryQubit,
        qm: QuantumModel,
        *,
        t: Time,
    ):
        super().__init__(t, f"addr={qubit.addr} key={qubit.key}")
        self.memory = memory
        self.qubit = qubit
        self.qm = qm

    @override
    def invoke(self) -> None:
        self.memory.handle(self)
