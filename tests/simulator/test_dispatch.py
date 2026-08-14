from typing import final, override

from mqns.simulator import Event, EventDispatcherMixin, Simulator, Time, event_handler


class OwnedEvent(Event):
    def __init__(self, t: Time, owner: "OwnerBase"):
        super().__init__(t)
        self.owner = owner

    @override
    def invoke(self):
        self.owner.handle(self)


@final
class EventA(OwnedEvent):
    pass


@final
class EventB(OwnedEvent):
    pass


@final
class EventC(OwnedEvent):
    pass


@final
class EventD(OwnedEvent):
    pass


class OwnerBase(EventDispatcherMixin):
    def __init__(self):
        self.invoked: list[str] = []

    @event_handler
    def handle_a(self, event: "EventA") -> None:
        _ = event
        self.invoked.append("B.a")

    @event_handler
    def handle_b(self, event: EventB) -> None:
        _ = event
        self.invoked.append("B.b")

    @event_handler
    def handle_c(self, event: EventC) -> None:
        _ = event
        self.invoked.append("B.c")


class OwnerSub(OwnerBase):
    @override
    @event_handler
    def handle_b(self, event: EventB) -> None:
        _ = event
        self.invoked.append("S.b")

    @override
    @event_handler
    def handle_c(self, event: EventC) -> None:
        event.cancel()
        self.invoked.append("S.c")

    @event_handler
    def handle_d(self, event: EventD) -> None:
        _ = event
        self.invoked.append("S.d")


def test_dispatcher():
    simulator = Simulator(0, 1, accuracy=1000)
    owner = OwnerSub()
    simulator.sched(EventA(simulator.time(slot=1), owner))
    simulator.sched(EventB(simulator.time(slot=2), owner))
    simulator.sched(EventC(simulator.time(slot=3), owner))
    simulator.sched(EventD(simulator.time(slot=4), owner))

    simulator.run()
    assert owner.invoked == ["B.a", "S.b", "B.b", "S.c", "S.d"]
