from typing import override

from mqns.entity.memory import MemoryQubit, PathDirection, QubitState
from mqns.entity.node import QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import FibPath
from mqns.network.fw.message import PathInstructions, validate_path_instructions
from mqns.network.fw.mux import MuxScheme
from mqns.utils import unwrap, unwrap_cast


class MuxSchemeReactive(MuxScheme):
    """
    ``MuxScheme`` adapted for reactive forwarding.
    """

    @override
    def validate_path_instructions(self, inst: PathInstructions) -> None:
        validate_path_instructions(inst, reactive=True)

    @override
    def install_path_adj(self, inst: PathInstructions, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = ch

        assert "reactive_qubits" in inst
        idx = fp.own_idx + (-1 if dir == PathDirection.L else 0)
        key = inst["reactive_qubits"][idx]

        try:
            qubit, _ = next(self.memory.find(lambda q, _: q.key == key))
        except StopIteration:
            raise ValueError(f"reactive_qubits[{idx}] refers to non-existent qubit {key}")

        qubit.path_id, qubit.path_direction = fp.path_id, dir
        addrs = [qubit.addr]
        self.fw.log_debug("allocating %s qubits: %s", dir, addrs)

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = fp, dir, ch
        raise ValueError(f"{self} should not receive PATH_DELETE command")

    @override
    def qubit_has_path_id(self) -> bool:
        return True

    @override
    def list_qubit_epr_path_ids(self, mq: MemoryQubit) -> list[int]:
        if mq.path_id is None:  # qubit not used in PATH_INSERT
            return []
        return [mq.path_id]

    @override
    def qubit_is_entangled(self, mq: MemoryQubit, epr: Entanglement, neighbor: QNode) -> FibPath | None:
        _ = epr, neighbor
        mq.state = QubitState.PURIF
        return self.fib.get_path(unwrap_cast(mq.path_id))

    @override
    def find_swap_with(self, mq0: MemoryQubit, epr0: Entanglement, fp: FibPath | None) -> tuple[MemoryQubit, FibPath] | None:
        _ = epr0
        try:
            mq1, _ = next(
                self.memory.find(
                    lambda q, _: (
                        self.qubits_swappable(mq0, q)  # basic condition met
                        and q.path_id == mq0.path_id  # allocated to the same path_id
                        and q.path_direction is not mq0.path_direction  # in the opposite path direction
                    ),
                    has=self.epr_type,
                )
            )
            return mq1, unwrap(fp)
        except StopIteration:
            return None
