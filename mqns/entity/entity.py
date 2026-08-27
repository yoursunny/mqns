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

from abc import ABC, abstractmethod

from mqns.simulator import Event, Simulator
from mqns.utils import LogSelfMixin


class Entity(LogSelfMixin, ABC):
    """
    Basic entity class.

    Examples of entities include memories, channels, and nodes.
    """

    def __init__(self, name: str):
        self.name = name
        """Entity name."""

    def install(self, simulator: Simulator) -> None:
        """
        Initialize the entity and schedule initial events.

        Args:
            simulator: An simulator that is not running.
        """
        assert not hasattr(self, "simulator") or self.simulator is simulator
        assert not simulator.running
        self.simulator = simulator
        """Global simulator instance."""

    @abstractmethod
    def handle(self, event: Event, /) -> None:
        """
        Process an event.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
