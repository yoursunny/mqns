import inspect
from collections.abc import Callable
from typing import Any, ClassVar, get_type_hints

from mqns.simulator.event import Event
from mqns.utils import DecoratorDispatchBuilder

type _HandlerFunc[E: Event] = Callable[[Any, E], bool | None]

_builder = DecoratorDispatchBuilder[type[Event], _HandlerFunc](
    classvar_name="_event_handlers",
    decorator_attr="_event_handler",
)


def _extract_event_type(f: _HandlerFunc) -> type[Event]:
    sig = inspect.signature(f)
    params = list(sig.parameters.values())
    if len(params) != 2:
        raise TypeError("@event_handler: handler must accept two parameters")

    hints = get_type_hints(f)
    typ = hints.get(params[1].name)
    if not (isinstance(typ, type) and issubclass(typ, Event)):
        raise TypeError("@event_handler: handler must accept Event subclass")
    if not getattr(typ, "__final__", False):
        raise TypeError(f"@event_handler: {typ} must be marked @final")
    return typ


def event_handler[E: Event](f: _HandlerFunc[E]) -> _HandlerFunc[E]:
    """
    Method decorator to register an event handler.

    Args:
        f: Handler function, ``def handle_event_a(self, event: EventA) -> bool|None``.

    This decorator must be used with ``EventDispatcherMixin``.
    See ``EventDispatcherMixin.handle`` for the semantics of ``f``'s return value.

    The handler registry is per class.
    Only one handler is allowed per event type, which could be registered from anywhere in the class hierarchy.
    If a subclass overrides a method, the overriding method must also have this decorator.
    """
    return _builder.make_decorator(_extract_event_type(f))(f)


class EventDispatcherMixin:
    """
    Mixin class for event dispatching functionality on event target (e.g. ``Application``).
    """

    __slots__ = ()
    _event_handlers: ClassVar[dict[type[Event], list[_HandlerFunc]]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._event_handlers = _builder.gather(cls)

    def handle(self, event: Event, /) -> bool | None:
        """
        Dispatch an event.

        Args:
            event: Event instance.

        Returns:

        * If True, the event is fully handled and not passed to the next event target.
        * Otherwise, the event is passed to the next event target.
        """
        handler = self._event_handlers.get(type(event))
        if handler:
            return handler[0](self, event)

        return False
