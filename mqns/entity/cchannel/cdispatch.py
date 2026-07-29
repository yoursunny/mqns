from collections.abc import Callable
from typing import Any, ClassVar

from mqns.entity.cchannel.cchannel import ClassicPacket, RecvClassicPacket
from mqns.simulator import event_handler
from mqns.utils import DecoratorDispatchBuilder

type _HandlerFunc = Callable[[Any, ClassicPacket, Any], Any]

_builder = DecoratorDispatchBuilder[str, _HandlerFunc](
    classvar_name="_classic_cmd_handlers",
    decorator_attr="_classic_cmd_handler",
)


def classic_cmd_handler(cmd: str):
    """
    Method decorator to indicate a classic command handler.

    Args:
        cmd: Command name, which should appear in ``cmd`` key of each message.
        f: Handler function, ``def handle_cmd(self, pkt: ClassicPacket, msg: dict) -> Any``.

    This decorator must be used with ``ClassicCommandDispatcherMixin``.

    The handler registry is per class.
    Only one handler is allowed per command name, which could be registered from anywhere in the class hierarchy.
    If a subclass overrides a method, the overriding method must also have this decorator.
    """
    return _builder.make_decorator(cmd)


class ClassicCommandModule:
    """
    Mixin class to indicate that the class may contain classic command handlers.

    This is effective only if the class appears as a member of ``ClassicCommandDispatcherMixin`` subclass.
    The member must be declared at the class level of the owning class,
    with a type hint that resolves to the subclass type.
    """

    __slots__ = ()
    _classic_cmd_handlers: ClassVar[dict[str, list[_HandlerFunc]]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._classic_cmd_handlers = _builder.gather(cls, ClassicCommandModule)


class ClassicCommandDispatcherMixin(ClassicCommandModule):
    """
    Provide classic packet dispatching functionality,
    assuming the classic packet contains JSON dict with "cmd" key.
    """

    __slots__ = ()

    @event_handler
    def handle_classic_command(self, event: RecvClassicPacket) -> bool:
        pkt = event.packet
        msg = pkt.get()

        if isinstance(msg, dict) and (cmd := msg.get("cmd")):
            handler = self._classic_cmd_handlers.get(cmd)
            if handler:
                return handler[0](self, pkt, msg)

        return False
