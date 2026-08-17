from mqns.entity.cchannel.cchannel import (
    ClassicChannel,
    ClassicChannelInitKwargs,
    ClassicPacket,
    RecvClassicPacket,
    extract_cchannel_args,
)
from mqns.entity.cchannel.cdispatch import ClassicCommandDispatcherMixin, ClassicCommandModule, classic_cmd_handler

__all__ = [
    "classic_cmd_handler",
    "ClassicChannel",
    "ClassicChannelInitKwargs",
    "ClassicCommandDispatcherMixin",
    "ClassicCommandModule",
    "ClassicPacket",
    "extract_cchannel_args",
    "RecvClassicPacket",
]

for name in __all__:
    globals()[name].__module__ = __name__
