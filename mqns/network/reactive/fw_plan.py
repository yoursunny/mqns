from typing import Final

from mqns.entity.memory import MemoryQubit
from mqns.network.fw import FibPath
from mqns.utils import unwrap_cast


class ReactivePlan:
    def __init__(self, fp: FibPath):
        self.fp: Final = fp


class ReactiveSwapPlan(ReactivePlan):
    mq1: MemoryQubit | None = None


class ReactiveConsumePlan(ReactivePlan):
    pass


class ReactivePlanner:
    """
    Forwarder data structure that tracks specific qubits in relation to FIB path entries.
    It stores the plan of what to do with each qubit when processing their ``QubitEntangledEvent``.
    """

    def __init__(self):
        self.swaps: dict[str, ReactiveSwapPlan] = {}
        self.consumes: dict[str, ReactiveConsumePlan] = {}

    def clear(self) -> None:
        self.swaps.clear()
        self.consumes.clear()

    def swap(self, fp: FibPath, key_l: str, key_r: str) -> None:
        """
        Save a plan to swap two qubits once they become ELIGIBLE.

        Args:
            fp: FIB path entry to associate the qubits with and to guide the swap.
            key_l: Qubit reservation key.
            key_r: Qubit reservation key.
        """
        plan = ReactiveSwapPlan(fp)
        self.swaps[key_l] = plan
        self.swaps[key_r] = plan

    def consume(self, fp: FibPath, key: str) -> None:
        """
        Save a plan to consume a qubit once it becomes ELIGIBLE.

        Args:
            fp: FIB path entry to associate the qubit with.
            key: Qubit reservation key.
        """
        self.consumes[key] = ReactiveConsumePlan(fp)

    def find_fib_path(self, mq: MemoryQubit) -> FibPath | None:
        """
        Find path assignment for a qubit.
        """
        key = unwrap_cast(mq.key)
        plan = self.consumes.pop(key, None) or self.swaps.get(key)

        if plan is None:
            # No plan to assign this qubit to any path.
            return None

        # Has plan that needs this qubit assigned to a path.
        return plan.fp

    def find_swap_with(self, mq0: MemoryQubit) -> tuple[MemoryQubit, FibPath] | None:
        """
        Find another qubit to swap with the given qubit.
        """
        key0 = unwrap_cast(mq0.key)
        plan = self.swaps.pop(key0, None)

        if plan is None:
            # No plan to swap this qubit.
            return None

        if plan.mq1 is None:
            # Has plan to swap. This is the first qubit becoming ELIGIBLE.
            # Store this qubit so it can be returned when the second qubit becomes ELIGIBLE.
            plan.mq1 = mq0
            return None

        # Has plan to swap. The other qubit has arrived and stored as plan.mq1.
        return plan.mq1, plan.fp
