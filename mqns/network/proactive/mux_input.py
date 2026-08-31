import copy
from collections.abc import Mapping
from typing import Literal

from mqns.network.proactive.mux import MuxScheme
from mqns.network.proactive.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.proactive.mux_dynamic_epr import MuxSchemeDynamicEpr
from mqns.network.proactive.mux_statistical import MuxSchemeStatistical

type MuxSchemeLiteral = Literal["B", "S", "D"]
"""
String identification of a ``MuxScheme`` constructor.
"""

MUX_SCHEME_MAP: Mapping[MuxSchemeLiteral, type[MuxScheme]] = {
    "B": MuxSchemeBufferSpace,
    "S": MuxSchemeStatistical,
    "D": MuxSchemeDynamicEpr,
}

type MuxSchemeInput = MuxScheme | MuxSchemeLiteral | None
"""
``MuxScheme`` input parsable by ``parse_mux_scheme``.
"""


def parse_mux_scheme(input: MuxSchemeInput) -> MuxScheme:
    """
    Parse a ``MuxScheme`` input.
    """
    if input is None:
        return MuxSchemeBufferSpace()
    if isinstance(input, MuxScheme):
        return copy.deepcopy(input)
    return MUX_SCHEME_MAP[input]()


def mux_scheme_is_buffer_space(input: MuxSchemeInput) -> bool:
    """
    Determine whether ``parse_mux_scheme(input)`` would return ``MuxSchemeBufferSpace``.
    """
    return input in (None, "B") or isinstance(input, MuxSchemeBufferSpace)
