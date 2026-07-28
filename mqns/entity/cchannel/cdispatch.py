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


def _populate_handlers(cls: type["ClassicCommandDispatcherMixin"]):
    # Identify handler method names.
    handler_names: dict[str, str] = {}
    for base in reversed(cls.mro()):
        for name, attr in base.__dict__.items():
            if cmd := getattr(attr, "_classic_cmd", None):
                handler_names[cmd] = name

    # Map commands to the most specific implementation in this class.
    cls._classic_handlers = {cmd: getattr(cls, name) for cmd, name in handler_names.items()}


class ClassicCommandDispatcherMixin:
    """
    Provide classic packet dispatching functionality,
    assuming the classic packet contains JSON dict with "cmd" key.
    """

    __slots__ = ()
    _classic_handlers: ClassVar[dict[str, _HandlerFunc]]

    @event_handler
    def handle_classic_command(self, event: RecvClassicPacket) -> bool:
        cls = type(self)

        if "_classic_handlers" not in cls.__dict__:
            _populate_handlers(cls)

        pkt = event.packet
        msg = pkt.get()
        if isinstance(msg, dict) and (cmd := msg.get("cmd")):
            handler = cls._classic_handlers.get(cmd)
            if handler:
                return handler(self, pkt, msg)

        return False
