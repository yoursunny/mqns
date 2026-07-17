from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import PathInstructions, validate_path_instructions
from mqns.network.fw.mux import MuxScheme
from mqns.network.fw.select import (
    MemoryEprIterator,
    MemoryEprTuple,
    call_select,
    select_random,
    select_swap_qubit_newest,
    select_swap_qubit_oldest,
)
from mqns.utils import log, unwrap_cast

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


class MuxSchemeFibBase(MuxScheme):
    class SelectSwapQubit(Protocol):
        """
        Function to select a preferred swap candidate.

        When qubit ``mq0`` on channel ``ch0`` becomes ELIGIBLE and the forwarder decides to swap it with a qubit from ``ch1``,
        but there are multiple candidate qubits from ``ch1``, this function is called to select a qubit.
        """

        def __call__(
            self, candidates: list[MemoryEprTuple], fw: "Forwarder", mq0: MemoryEprTuple, fib_entry: FibEntry, /
        ) -> MemoryEprTuple:
            """
            Args:
                candidates: List of qubits from ``ch1``, guaranteed to have two or more items.
                fw: The forwarder instance.
                mq0: The qubit from ``ch0``.
                fib_entry: The FIB entry.

            Returns:
                One item from ``candidates``.
            """
            ...

    SelectSwapQubit_random: SelectSwapQubit = select_random
    """
    Select a random qubit among the candidates, following uniform distribution.
    """
    SelectSwapQubit_oldest: SelectSwapQubit = select_swap_qubit_oldest
    """
    Select the qubit that becomes ELIGIBLE in this forwarder the earliest, i.e. First-In-First-Out (FIFO).
    """
    SelectSwapQubit_newest: SelectSwapQubit = select_swap_qubit_newest
    """
    Select the qubit that becomes ELIGIBLE in this forwarder the latest, i.e. Last-In-First-Out (LIFO).
    """

    def __init__(self, name: str, select_swap_qubit: SelectSwapQubit | None):
        super().__init__(name)
        self._select_swap_qubit = select_swap_qubit

    @override
    def find_swap_candidate(
        self, mq0: MemoryQubit, epr0: Entanglement, fib_entry: FibEntry | None, input: MemoryEprIterator
    ) -> tuple[MemoryQubit, FibEntry] | None:
        assert fib_entry
        mt1 = call_select(
            self.list_swap_candidates(mq0, fib_entry, input), self._select_swap_qubit, self.fw, (mq0, epr0), fib_entry
        )
        return None if mt1 is None else (mt1[0], fib_entry)

    @abstractmethod
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator) -> MemoryEprIterator: ...


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    """
    Buffer-Space multiplexing scheme.
    """

    def __init__(
        self,
        name="buffer-space multiplexing",
        *,
        select_swap_qubit: MuxSchemeFibBase.SelectSwapQubit | None = None,
    ):
        """
        Args:
            select_swap_qubit: Function to select a qubit to swap with, default is first.
        """
        super().__init__(name, select_swap_qubit)

    @override
    def validate_path_instructions(self, instructions: PathInstructions) -> None:
        validate_path_instructions(instructions)
        assert "m_v" in instructions

    @override
    def install_path_adj(
        self,
        instructions: PathInstructions,
        fib_entry: FibEntry,
        direction: PathDirection,
        qchannel: QuantumChannel,
    ) -> None:
        assert "m_v" in instructions
        mv = instructions["m_v"]
        mv_offset, ch_side = (-1, 1) if direction == PathDirection.L else (0, 0)
        mv_index = fib_entry.own_idx + mv_offset
        mv_element = mv[mv_index]

        if isinstance(mv_element, str):
            # allocate a specific memory qubit identified with reservation key (only used in reactive forwarding)
            qubit, _ = next(self.memory.find(lambda q, _: q.key == mv_element), (None, None))
            if qubit is None:
                raise ValueError(f"m_v[{mv_index}] refers to non-existent qubit {mv_element}")
            qubit.path_id, qubit.path_direction = fib_entry.path_id, direction
            addrs = [qubit.addr]
        else:
            # allocate memory qubit(s) assigned to the channel (typically used in proactive forwarding)
            n_qubits = mv_element[ch_side]
            addrs = self.memory.allocate(
                qchannel,
                fib_entry.path_id,
                direction,
                n="all" if n_qubits == 0 else n_qubits,
            )
        log.debug(f"{self.fw}: allocated {direction} qubits: {addrs}")

    @override
    def uninstall_path_adj(self, fib_entry: FibEntry, direction: PathDirection, qchannel: QuantumChannel) -> None:
        qubits = self.memory.find(lambda q, _: q.path_id == fib_entry.path_id, qchannel=qchannel)
        addrs = [q[0].addr for q in qubits]
        self.memory.deallocate(*addrs)
        log.debug(f"{self.fw}: deallocated {direction} qubits: {addrs}")
        pass

    @override
    def qubit_has_path_id(self) -> bool:
        return True

    @override
    def list_qubit_epr_path_ids(self, mq: MemoryQubit) -> list[int]:
        if mq.path_id is None:
            return []
        return [mq.path_id]

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement, neighbor: QNode) -> FibEntry | None:
        _ = epr, neighbor
        mq.state = QubitState.PURIF
        return self.fib.get(unwrap_cast(mq.path_id))

    @override
    def list_swap_candidates(self, mq0: MemoryQubit, fib_entry: FibEntry, input: MemoryEprIterator):
        return (
            (q, v)
            for (q, v) in input
            if q.path_id == fib_entry.path_id  # allocated to the same path_id
            and q.path_direction is not mq0.path_direction  # in the opposite path direction
        )
