from abc import abstractmethod

from qns.entity.memory import MemoryQubit, PathDirection, QubitState
from qns.entity.node import QNode
from qns.entity.qchannel import QuantumChannel
from qns.models.epr import WernerStateEntanglement
from qns.network.proactive.fib import FIBEntry, is_swap_disabled
from qns.network.proactive.message import InstallPathInstructions
from qns.network.proactive.mux import MuxScheme
from qns.utils import log

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class MuxSchemeFibBase(MuxScheme):
    @override
    def qubit_is_eligible(self, qubit: MemoryQubit, fib_entry: FIBEntry | None) -> None:
        assert fib_entry is not None
        assert qubit.qchannel is not None

        if is_swap_disabled(fib_entry):  # no swapping in isolated links
            self.fw.consume_and_release(qubit)
            return

        if fib_entry.own_idx in (0, len(fib_entry.route) - 1):  # this is an end node
            self.fw.consume_and_release(qubit)
            return

        # this is an intermediate node
        # look for another eligible qubit
        mq1 = self.select_eligible_qubit(qubit, fib_entry)
        if mq1:  # do swapping
            self.fw.do_swapping(qubit, mq1, fib_entry, fib_entry)

    @abstractmethod
    def select_eligible_qubit(self, mq0: MemoryQubit, fib_entry: FIBEntry) -> MemoryQubit | None:
        pass


class MuxSchemeBufferSpace(MuxSchemeFibBase):
    def __init__(self, name="buffer-space multiplexing"):
        super().__init__(name)

    @override
    def validate_path_instructions(self, instructions: InstallPathInstructions) -> None:
        assert instructions["mux"] == "B"
        assert "m_v" in instructions
        assert len(instructions["m_v"]) + 1 == len(instructions["route"])

    @override
    def install_path_neighbor(
        self,
        instructions: InstallPathInstructions,
        fib_entry: FIBEntry,
        direction: PathDirection,
        neighbor: QNode,
        qchannel: QuantumChannel,
    ) -> None:
        assert "m_v" in instructions
        m_v = instructions["m_v"]
        m_v_offset, ch_side = (-1, 1) if direction == PathDirection.LEFT else (0, 0)

        n_qubits = m_v[fib_entry.own_idx + m_v_offset][ch_side]
        if n_qubits == 0:  # 0 means use all qubits assigned to this qchannel
            n_qubits = len(self.memory.get_channel_qubits(qchannel.name))

        qubits = self.memory.allocate(fib_entry.path_id, direction, ch_name=qchannel.name, n=n_qubits)
        log.debug(f"{self.own}: Allocated {direction} qubits: {qubits}")

    @override
    def qubit_is_entangled(self, qubit: MemoryQubit, neighbor: QNode) -> None:
        assert qubit.path_id is not None
        fib_entry = self.fib.get(qubit.path_id)
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, fib_entry, neighbor)

    @override
    def select_eligible_qubit(self, mq0: MemoryQubit, fib_entry: FIBEntry) -> MemoryQubit | None:
        assert mq0.path_id is not None
        possible_path_ids = {fib_entry.path_id}
        if not self.fw.isolate_paths:
            # if not isolated paths -> include other paths serving the same request
            possible_path_ids = self.fib.list_path_ids_by_request_id(fib_entry.req_id)
            log.debug(f"{self.own}: path ids {possible_path_ids}")

        mq1, _ = next(
            self.memory.find(
                lambda q, _: q.state == QubitState.ELIGIBLE  # in ELIGIBLE state
                and q.qchannel != mq0.qchannel  # assigned to a different channel
                and q.path_id in possible_path_ids  # allocated to the same path_id or another path_id under the same request_id
                and q.path_direction != mq0.path_direction,  # in the opposite path direction
                has_epr=True,
            ),
            (None, None),
        )
        # TODO selection algorithm among found qubits
        return mq1

    @override
    def swapping_succeeded(
        self,
        prev_epr: WernerStateEntanglement,
        next_epr: WernerStateEntanglement,
        new_epr: WernerStateEntanglement,
    ) -> None:
        assert prev_epr.tmp_path_ids is None
        assert next_epr.tmp_path_ids is None
        _ = new_epr

    @override
    def su_parallel_avoid_conflict(self, my_new_epr: WernerStateEntanglement, su_path_id: int) -> bool:
        assert my_new_epr.tmp_path_ids is None
        _ = su_path_id
        return False

    @override
    def su_parallel_succeeded(
        self, merged_epr: WernerStateEntanglement, new_epr: WernerStateEntanglement, other_epr: WernerStateEntanglement
    ) -> None:
        assert new_epr.tmp_path_ids is None
        assert other_epr.tmp_path_ids is None
        _ = merged_epr
