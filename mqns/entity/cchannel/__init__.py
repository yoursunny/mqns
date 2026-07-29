from mqns.entity.cchannel.cchannel import ClassicChannel, ClassicChannelInitKwargs, ClassicPacket, RecvClassicPacket
from mqns.entity.cchannel.cdispatch import ClassicCommandDispatcherMixin, ClassicCommandModule, classic_cmd_handler

__all__ = [
    "classic_cmd_handler",
    "ClassicChannel",
    "ClassicChannelInitKwargs",
    "ClassicCommandDispatcherMixin",
    "ClassicCommandModule",
    "ClassicPacket",
    "RecvClassicPacket",
]

for name in __all__:
    globals()[name].__module__ = __name__
