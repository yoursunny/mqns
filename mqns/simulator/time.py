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

import functools
from typing import Final, final

_ACCURACY0_STRS: dict[int, str] = {
    0: "(sentinel)",
    -1: "(min)",
    1: "(max)",
}


@final
@functools.total_ordering
class Time:
    """
    Timestamp or duration used in the simulator.
    """

    __slots__ = ("time_slot", "accuracy")

    SENTINEL: "Time"
    """Invalid Time instance as placeholder."""
    MIN: "Time"
    """Minimum value, compares less than all other instances."""
    MAX: "Time"
    """Maximum value, compares greater than all other instances."""

    def __init__(self, time_slot: int, *, accuracy: int):
        """
        Construct from time slot.

        Args:
            time_slot: Integer time slot.
            accuracy: How many time slots per second.
        """
        self.time_slot: Final[int] = time_slot
        self.accuracy: Final[int] = accuracy

    @staticmethod
    def sec_to_time_slot(sec: float, accuracy: int) -> int:
        """
        Convert seconds to time slots at given accuracy.
        """
        return round(sec * accuracy)

    @staticmethod
    def from_sec(sec: float, *, accuracy: int) -> "Time":
        """
        Construct Time from seconds.

        Args:
            sec: seconds.
            accuracy: how many time slots per second.
        """
        return Time(Time.sec_to_time_slot(sec, accuracy), accuracy=accuracy)

    @property
    def sec(self) -> float:
        """
        Retrieve timestamp/duration in seconds.
        """
        return self.time_slot / self.accuracy

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison operator.

        Time instance is equal to the other object only if:

        * The other object is also a Time instance.
        * They have the same accuracy.
        * They have the same time slot.
        """
        return type(other) is Time and self.accuracy == other.accuracy and self.time_slot == other.time_slot

    def __lt__(self, other: "Time") -> bool:
        """
        Less than comparison operator.
        Two Time instances can be compared only if they have the same accuracy.
        """
        if self.accuracy == other.accuracy:
            return self.time_slot < other.time_slot
        if self.accuracy == 0:
            return self.time_slot < 0
        if other.accuracy == 0:
            return other.time_slot > 0
        raise ValueError("cannot compare Time with different accuracy")

    def __hash__(self) -> int:
        return hash(self.time_slot)

    def __add__(self, ts: "Time|float") -> "Time":
        """
        Add a duration and returns a new Time object.

        Args:
            ts: either a Time object with same accuracy, or a duration number in seconds.
        """
        if type(ts) is Time:
            assert ts.accuracy == self.accuracy
            time_slot = ts.time_slot
        else:
            time_slot = Time.sec_to_time_slot(ts, self.accuracy)
        return Time(time_slot=self.time_slot + time_slot, accuracy=self.accuracy)

    def __sub__(self, ts: "Time|float") -> "Time":
        """
        Subtract a duration and returns a new Time object.

        Args:
            ts: either a Time object with same accuracy, or a duration number in seconds.
        """
        if type(ts) is Time:
            assert ts.accuracy == self.accuracy
            time_slot = ts.time_slot
        else:
            time_slot = Time.sec_to_time_slot(ts, self.accuracy)
        return Time(time_slot=self.time_slot - time_slot, accuracy=self.accuracy)

    def __repr__(self) -> str:
        return _ACCURACY0_STRS[self.time_slot] if self.accuracy == 0 else str(self.sec)


Time.SENTINEL = Time(0, accuracy=0)
Time.MIN = Time(-1, accuracy=0)
Time.MAX = Time(1, accuracy=0)
