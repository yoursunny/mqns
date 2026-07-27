from collections.abc import Callable
from typing import Any, Unpack

from mqns.entity.cchannel import ClassicChannel, ClassicChannelInitKwargs, ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.models.delay import UniformDelayModel
from mqns.simulator import Simulator, event_handler, func_to_event


class ClassicApp(Application[Node]):
    def __init__(self):
        super().__init__()

        self.n_tx = 0
        self.rx: list[RecvClassicPacket] = []

    @event_handler
    def handle_recv(self, event: RecvClassicPacket) -> None:
        self.rx.append(event)

    def send(self, dest: Node, msgs: list[Any]):
        for msg in msgs:
            self.n_tx += 1
            self.node.send_cpacket(dest, ClassicPacket(msg, src=self.node, dest=dest))

    def sched_sends(self, n: int, dest: Node, since: float, interval: float, msgs: Callable[[int], list[Any]] = lambda _: [""]):
        t = self.simulator.time(sec=since)
        for i in range(n):
            event = func_to_event(t, self.send, dest, list(msgs(i)))
            self.simulator.add_event(event)
            t += interval


def make_nodes(n=2, **kwargs: Unpack[ClassicChannelInitKwargs]):
    cc = ClassicChannel("cc", **kwargs)
    nodes: list[Node] = []
    apps: list[ClassicApp] = []
    for i in range(n):
        node = Node(f"n{i}")
        node.add_cchannel(cc)
        app = ClassicApp()
        node.add_apps(app)
        nodes.append(node)
        apps.append(app)
    simulator = Simulator(0, 10, accuracy=1000, install_to=nodes)
    return simulator, *apps


def test_delay_const():
    simulator, a0, a1 = make_nodes(delay=0.2)
    a0.sched_sends(1000, a1.node, 1.000, 0.001, lambda i: [{"A": i}])
    simulator.run()

    assert a0.n_tx == 1000
    assert len(a1.rx) == 1000
    for i, event in enumerate(a1.rx):
        assert event.t == simulator.time(sec=1.200 + i / 1000)
        assert event.packet.get() == {"A": i}


def test_delay_uniform():
    simulator, a0, a1 = make_nodes(delay=UniformDelayModel(min=0.1, max=0.5))
    a0.sched_sends(1000, a1.node, 1.000, 0.000)
    simulator.run()

    assert a0.n_tx == 1000
    assert len(a1.rx) == 1000

    min_t = min(event.t for event in a1.rx)
    max_t = max(event.t for event in a1.rx)
    assert 1.100 <= min_t.sec < 1.150
    assert 1.450 < max_t.sec <= 1.500


def test_drop():
    simulator, a0, a1 = make_nodes(drop_rate=0.3)
    a0.sched_sends(1000, a1.node, 1.000, 0.001)
    simulator.run()

    assert a0.n_tx == 1000
    assert 500 < len(a1.rx) < 900


def test_bandwidth():
    # 4 octets per packet, 1 packets allowed per 0.010 seconds, 2 packets can be buffered
    simulator, a0, a1 = make_nodes(bandwidth=400, max_buffer_size=8)
    # sending 100 packets in 0.1 seconds, 10 are sent on time, 2 are buffered
    a0.sched_sends(100, a1.node, 1.000, 0.001, lambda i: [f"{i:04}"])
    simulator.run()

    assert a0.n_tx == 100
    assert [(int(event.t.sec * 1000), event.packet.get()) for event in a1.rx] == [
        (1000, "0000"),  # sent on time
        (1010, "0001"),  # queued
        (1020, "0002"),  # queued
        (1030, "0010"),  # admitted into queue when "0001" is sent
        (1040, "0020"),  # admitted into queue when "0002" is sent
        (1050, "0030"),
        (1060, "0040"),
        (1070, "0050"),
        (1080, "0060"),
        (1090, "0070"),
        (1100, "0080"),
        (1110, "0090"),
    ]
    assert len(a1.rx) == 12
