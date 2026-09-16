import copy
from collections.abc import Mapping
from typing import Literal

from mqns.entity.qchannel.link_arch import LinkArch
from mqns.entity.qchannel.link_arch_dim import LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual
from mqns.entity.qchannel.link_arch_sim import LinkArchSim
from mqns.entity.qchannel.link_arch_sr import LinkArchSr

type LinkArchLiteral = Literal["DIM-BK", "DIM-BK-SeQUeNCe", "DIM-dual", "SR", "SIM"]
"""
String representation of commonly used link architectures.
"""

LINK_ARCH_MAP: Mapping[LinkArchLiteral, type[LinkArch]] = {
    "DIM-BK": LinkArchDimBk,
    "DIM-BK-SeQUeNCe": LinkArchDimBkSeq,
    "DIM-dual": LinkArchDimDual,
    "SR": LinkArchSr,
    "SIM": LinkArchSim,
}

type LinkArchInput = LinkArch | type[LinkArch] | LinkArchLiteral | None
"""
``LinkArch`` input parsable by ``parse_link_arch``.

* ``LinkArch`` subclass constructor: construct new instance.
* ``LinkArch`` instance: deep-copied.
* ``None``: ``LinkArchDimBkSeq``.
* Literal string: Lookup ``LINK_ARCH_MAP``.
"""


def parse_link_arch(input: LinkArchInput) -> LinkArch:
    """
    Parse a ``LinkArch`` input.
    """
    if input is None:
        return LinkArchDimBkSeq()

    if isinstance(input, str):
        return LINK_ARCH_MAP[input]()

    if callable(input):
        return input()

    return copy.deepcopy(input)
