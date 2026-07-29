from mqns.entity.qchannel.link_arch import LinkArch, LinkArchAlways, LinkArchParameters
from mqns.entity.qchannel.link_arch_dim import LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual
from mqns.entity.qchannel.link_arch_sim import LinkArchSim
from mqns.entity.qchannel.link_arch_sr import LinkArchSr
from mqns.entity.qchannel.qchannel import QuantumChannel, QuantumChannelInitKwargs, RecvQubitPacket

__all__ = [
    "LinkArch",
    "LinkArchAlways",
    "LinkArchDimBk",
    "LinkArchDimBkSeq",
    "LinkArchDimDual",
    "LinkArchParameters",
    "LinkArchSim",
    "LinkArchSr",
    "QuantumChannel",
    "QuantumChannelInitKwargs",
    "RecvQubitPacket",
]

for name in __all__:
    globals()[name].__module__ = __name__
