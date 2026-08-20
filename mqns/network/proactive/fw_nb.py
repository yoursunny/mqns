from typing import override

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw import FibPath, ForwarderNorthbound
from mqns.network.protocol.event import PathActivateEvent, PathDeactivateEvent


class ProactiveForwarderNorthbound(ForwarderNorthbound):
    @override
    def install_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        self.simulator.sched(
            PathActivateEvent(
                self.node,
                ch,
                self._ll_path_id(fp),
                t=self.simulator.tc,
                is_primary=dir is PathDirection.R,
            )
        )

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = dir
        self.simulator.sched(
            PathDeactivateEvent(
                self.node,
                ch,
                self._ll_path_id(fp),
                t=self.simulator.tc,
            )
        )

    def _ll_path_id(self, fp: FibPath) -> int | None:
        return fp.path_id if self.mux.qubit_has_path_id() else None
