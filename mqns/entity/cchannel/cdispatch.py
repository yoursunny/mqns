from collections.abc import Callable
from typing import Any, ClassVar

from mqns.entity.cchannel.cchannel import ClassicPacket, RecvClassicPacket
from mqns.simulator import event_handler

type _HandlerFunc = Callable[[Any, ClassicPacket, Any], Any]


def classic_cmd_handler(cmd: str):
    """
    Method decorator to indicate a classic command handler.
    """

    def decorator(f: _HandlerFunc):
        setattr(f, "_classic_cmd", cmd)
        return f

    return decorator


def _make_wrapper(attr_name: str, handler: Any):
    def wrapper(self: Any, pkt: ClassicPacket, msg: Any):
        return handler(getattr(self, attr_name), pkt, msg)

    return wrapper


class ClassicCommandModule:
    """
    Indicate that classic command handlers may be declared within a subclass of this type
    when it appears as a member of ``ClassicCommandDispatcherMixin`` subclass.

    The member must be declared at the class level of the owning class,
    with a type hint that resolves to the subclass type.
    """

    __slots__ = ()
    _classic_handlers: ClassVar[dict[str, _HandlerFunc]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        handlers: dict[str, _HandlerFunc] = {}

        for base in reversed(cls.mro()):
            handlers.update(base.__dict__.get("_classic_handlers", {}))

        for attr_name, attr_type in cls.__annotations__.items():
            if isinstance(attr_type, type) and issubclass(attr_type, ClassicCommandModule):
                for cmd, handler in attr_type._classic_handlers.items():
                    handlers[cmd] = _make_wrapper(attr_name, handler)

        for attr in cls.__dict__.values():
            if cmd := getattr(attr, "_classic_cmd", None):
                handlers[cmd] = attr

        cls._classic_handlers = handlers


class ClassicCommandDispatcherMixin(ClassicCommandModule):
    """
    Provide classic packet dispatching functionality,
    assuming the classic packet contains JSON dict with "cmd" key.
    """

    __slots__ = ()

    @event_handler
    def handle_classic_command(self, event: RecvClassicPacket) -> bool:
        cls = type(self)

        pkt = event.packet
        msg = pkt.get()
        if isinstance(msg, dict) and (cmd := msg.get("cmd")):
            handler = cls._classic_handlers.get(cmd)
            if handler:
                return handler(self, pkt, msg)

        return False
