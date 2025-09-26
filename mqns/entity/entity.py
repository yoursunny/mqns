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


class Entity(ABC):
    """
    Basic entity class.

    Examples of entities include memories, channels, and nodes.
    """

    def __init__(self, name: str):
        """
        Args:
            name: the name of this entity.

        """
        self.name = name
        self._simulator: Simulator | None = None

    @property
    def simulator(self) -> Simulator:
        """
        Return the Simulator that this entity belongs to.

        Raises:
            IndexError - simulator does not exist
        """
        if self._simulator is None:
            raise IndexError(f"{self} is not in a simulator")
        return self._simulator

    def install(self, simulator: Simulator) -> None:
        """
        Initialize the entity and schedule initial events.
        This must be invoked before `simulator.run()`.

        Args:
            simulator: the simulator.

        """
        assert self._simulator is None or self._simulator == simulator
        assert not simulator.running
        self._simulator = simulator

    @abstractmethod
    def handle(self, event: Event) -> None:
        """
        Process a received event.

        Args:
            event: the event that targets this entity.

        """
        pass

    def __repr__(self) -> str:
        return f"<entity {self.name}>"
