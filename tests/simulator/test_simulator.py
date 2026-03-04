import math
import threading
import time
from collections import defaultdict
from typing import override

import pytest

from mqns.simulator import Event, Simulator, Time


class SimpleEvent(Event):
    invoke_count = defaultdict[str, int](lambda: 0)

    @override
    def invoke(self) -> None:
        self.invoke_count[self.name or ""] += 1


class StopEvent(SimpleEvent):
    def __init__(self, t: Time, name: str, simulator: Simulator):
        super().__init__(t, name)
        self.simulator = simulator

    def invoke(self) -> None:
        super().invoke()
        assert self.simulator.running
        self.simulator.stop()
        assert not self.simulator.running


@pytest.fixture(autouse=True)
def clear_invoke_count():
    yield
    SimpleEvent.invoke_count.clear()


def test_run():
    s = Simulator(0, 15, accuracy=1000)
    assert s.total_events == 0

    e = SimpleEvent(s.time(sec=1), name="t0")
    s.add_event(e)
    e.cancel()
    # 1 instance of t0 scheduled at 1.0 but will not be invoked
    assert s.total_events == 1

    t = 0
    while t <= 12:
        s.add_event(SimpleEvent(s.time(sec=t), name="t1"))
        t += 0.5
    # 25 instances of t1 scheduled at 0.0, 0.5, 1.0, .., 11.5, 12.0
    assert s.total_events == 1 + 25

    t = 5
    while t <= 20:
        s.add_event(SimpleEvent(s.time(sec=t), name="t2"))
        t += 1
    # 11 instances of t2 scheduled at 5, 6, .., 14, 15
    assert s.total_events == 1 + 25 + 11

    assert not s.running
    s.run()
    assert s.tc == s.te
    assert not s.running

    assert SimpleEvent.invoke_count["t0"] == 0
    assert SimpleEvent.invoke_count["t1"] == 25
    assert SimpleEvent.invoke_count["t2"] == 11


@pytest.mark.parametrize(
    "te",
    [
        15,
        math.inf,
    ],
)
def test_stop(*, te: float):
    s = Simulator(0, te, accuracy=1000)
    s.update_gate(s.time(sec=60))

    e = StopEvent(s.time(sec=9.5), name="s0", simulator=s)
    s.add_event(e)
    # 1 instance of s0 scheduled at 9.5
    assert s.total_events == 1

    t = 1
    while t <= 60:
        e = SimpleEvent(s.time(sec=t), name="t1")
        s.add_event(e)
        t += 1
    # up to 60 instances of t1 scheduled at 1, 2, .., MIN(60, te)
    assert s.total_events == 1 + min(60, te)

    s.run()
    assert s.tc.sec < te

    assert SimpleEvent.invoke_count["t1"] == 9
    assert SimpleEvent.invoke_count["s0"] == 1


def test_gate():
    s = Simulator(0.1, math.inf, accuracy=1000)
    assert s.tc == s.ts == s.time(sec=0.1)
    s.add_event(SimpleEvent(s.time(sec=0), "z0"))  # before s.ts, dropped

    # set initial gate to 5s, schedule events
    s.update_gate(s.time(sec=5))
    s.add_event(SimpleEvent(s.time(sec=2), "b2"))
    s.add_event(SimpleEvent(s.time(sec=5), "b5"))
    s.add_event(SimpleEvent(s.time(sec=8), "a8"))

    # run Simulator in a background thread
    th = threading.Thread(target=s.run, daemon=True)
    th.start()

    # let the Simulator hit the gate, verify b2,b5 invoked
    time.sleep(0.2)
    assert SimpleEvent.invoke_count["b2"] == 1
    assert SimpleEvent.invoke_count["b5"] == 1
    assert SimpleEvent.invoke_count["a8"] == 0
    assert s.tc == s.time(sec=5)
    assert th.is_alive()

    # schedule another event at initial gate and release the gate to 10s
    s.add_event(SimpleEvent(s.time(sec=4), "z4"))  # before s.tc, dropped
    s.add_event(SimpleEvent(s.time(sec=5), "a5"))

    # let the Simulator run again, verify a5,a8 invoked
    s.update_gate(s.time(sec=10))
    time.sleep(0.2)
    assert SimpleEvent.invoke_count["a5"] == 1
    assert SimpleEvent.invoke_count["a8"] == 1
    assert s.tc == s.time(sec=8)
    assert th.is_alive()

    # verify dropped events are not invoked
    assert SimpleEvent.invoke_count["z0"] == 0
    assert SimpleEvent.invoke_count["z4"] == 0
    assert s.total_events == 4

    # stop and cleanup
    s.stop()
    th.join(timeout=1)
    assert not th.is_alive()
