from typing import TYPE_CHECKING

from mqns.entity.memory import QuantumMemory
from mqns.entity.node import Application, QNode
from mqns.models.epr import Entanglement
from mqns.network.fw.fib import Fib
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator
from mqns.utils import LogSelfMixin

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder, ForwarderCounters


class ForwarderModule(LogSelfMixin):
    """
    Base class of a module within the forwarder.
    """

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory
    fib: Fib
    fw_cnt: "ForwarderCounters"

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory
        self.fib = fw.fib
        self.fw_cnt = fw.cnt

    def __repr__(self):
        return Application.__repr__(self)
