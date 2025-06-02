from typing import Any

from qns.entity.memory import QuantumMemory
from qns.entity.node import Application, QNode
from qns.entity.qchannel import QuantumChannel, QubitLossChannel, RecvQubitPacket
from qns.models.qubit import Qubit
from qns.simulator import Event, Simulator, Time, func_to_event


class QNodeWithMemory(QNode):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.set_memory(QuantumMemory(capacity=1, decoherence_rate=0))


class QuantumRecvNode(QNodeWithMemory):
    def handle(self, event: Event) -> None:
        if isinstance(event, RecvQubitPacket):
            print(event.t, event.qubit)


class QuantumSendNode(QNodeWithMemory):
    def __init__(self, name: str, dest: QNode):
        super().__init__(name=name)
        self.dest = dest

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        assert self._simulator is not None

        t = 0
        while t < 10:
            time = self._simulator.time(sec=t)
            event = SendEvent(time, node=self, by=self)
            self._simulator.add_event(event)
            t += 0.25

    def send(self):
        assert self._simulator is not None
        print(self._simulator.current_time, "send qubit")
        link: QuantumChannel = self.qchannels[0]
        dest = self.dest
        qubit = Qubit()
        link.send(qubit, dest)


class SendEvent(Event):
    def __init__(self, t: Time, *, name: str|None = None, node: QuantumSendNode, by: Any = None):
        super().__init__(t=t, name=name, by=by)
        self.node: QuantumSendNode = node

    def invoke(self) -> None:
        self.node.send()


def test_qchannel_first():
    n2 = QuantumRecvNode("n2")
    n1 = QuantumSendNode("n1", dest=n2)
    l1 = QuantumChannel(name="l1", bandwidth=3, delay=0.2, max_buffer_size=5)
    # l2 = QuantumChannel(name="l2", bandwidth=5, delay=0.5, max_buffer_size=5)
    n1.add_qchannel(l1)
    n2.add_qchannel(l1)
    s = Simulator(0, 10, 1000)
    n1.install(s)
    n2.install(s)
    s.run()


class SendApp(Application):
    def __init__(self, dest: QNode, qchannel: QuantumChannel, send_rate=1):
        super().__init__()
        self.dest = dest
        self.qchannel = qchannel
        self.send_rate = send_rate

    def install(self, node, simulator: Simulator):
        super().install(node=node, simulator=simulator)
        assert self._simulator is not None
        t = self._simulator.ts
        event = func_to_event(t, self.send, by=self)
        self._simulator.add_event(event)

    def send(self):
        assert self._simulator is not None
        qubit = Qubit()
        self.qchannel.send(qubit=qubit, next_hop=self.dest)
        t = self._simulator.current_time + self._simulator.time(sec=1 / self.send_rate)
        event = func_to_event(t, self.send, by=self)
        self._simulator.add_event(event)


class RecvApp(Application):
    def __init__(self):
        super().__init__()
        self.add_handler(self.RecvQubitHandler, [RecvQubitPacket])

    def RecvQubitHandler(self, node, event: Event) -> bool|None:
        recv_time = event.t
        print("recv_time:{}".format(recv_time))


def test_qchannel_second():
    n1 = QNodeWithMemory(name="n_1")
    n2 = QNodeWithMemory(name="n_2")
    l1 = QuantumChannel(name="l_1")
    # l2 = QuantumChannel(name="l2", bandwidth=5, delay=0.5, drop_rate=0.2, max_buffer_size=5)
    n1.add_qchannel(l1)
    n2.add_qchannel(l1)
    s = Simulator(1, 5, 1000)
    n1.add_apps(SendApp(dest=n2, qchannel=l1))
    n2.add_apps(RecvApp())
    n1.install(s)
    n2.install(s)
    s.run()


def test_qubit_loss_channel():
    n1 = QNodeWithMemory(name="n_1")
    n2 = QNodeWithMemory(name="n_2")
    l1 = QubitLossChannel(name="loss_channel_1", p_init=0.1, attenuation_rate=0.02, length=100)
    print(l1.drop_rate)
    n1.add_qchannel(l1)
    n2.add_qchannel(l1)
    s = Simulator(1, 5, 1000)
    n1.add_apps(SendApp(dest=n2, qchannel=l1))
    n2.add_apps(RecvApp())
    n1.install(s)
    n2.install(s)
    s.run()
