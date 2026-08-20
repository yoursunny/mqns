from typing import override

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw import FibPath, ForwarderNorthbound
from mqns.network.reactive.message import LinkStateEntry, LinkStateMsg


class ReactiveForwarderNorthbound(ForwarderNorthbound):
    @override
    def install_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = dir, ch
        if not self.node.timing.is_routing():
            self.log_warning(
                "received INSTALL_PATH message for path %s outside of ROUTING phase; t_rtg is too short?", fp.path_id
            )

    @override
    def uninstall_path_adj(self, fp: FibPath, dir: PathDirection, ch: QuantumChannel) -> None:
        _ = fp, dir, ch
        raise ValueError(f"{self} should not receive PATH_DELETE command")

    def send_link_state(self):
        """
        Send link state message to controller. Assumes direct connection to controller.
        """
        link_states: list[LinkStateEntry] = []
        for event in self.fw.waiting_etg:
            assert event.qubit.key is not None
            link_states.append({"node": event.node.name, "neighbor": event.neighbor.name, "qubit": event.qubit.key})

        if len(link_states) == 0:
            self.log_debug("no link_state to send")
            return
        else:
            self.log_debug("send link_state for %s etg qubits", len(self.fw.waiting_etg))

        msg: LinkStateMsg = {
            "cmd": "LS",
            "ls": link_states,
        }
        self.send_ctrl(msg)
