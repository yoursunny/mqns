import copy
from collections.abc import Mapping
from typing import Literal

from mqns.network.fw import MuxScheme
from mqns.network.proactive.mux_buffer_space import MuxSchemeBufferSpace
from mqns.network.proactive.mux_dynamic_epr import MuxSchemeDynamicEpr
from mqns.network.proactive.mux_statistical import MuxSchemeStatistical

type MuxSchemeLiteral = Literal["B", "S", "D"]

_MUX_SCHEME_NAMED: Mapping[MuxSchemeLiteral, type[MuxScheme]] = {
    "B": MuxSchemeBufferSpace,
    "S": MuxSchemeStatistical,
    "D": MuxSchemeDynamicEpr,
}

type MuxSchemeInput = MuxScheme | MuxSchemeLiteral | None


def parse_mux_scheme(input: MuxSchemeInput) -> MuxScheme:
    if input is None:
        return MuxSchemeBufferSpace()
    if isinstance(input, MuxScheme):
        return copy.deepcopy(input)
    return _MUX_SCHEME_NAMED[input]()


def mux_scheme_is_buffer_space(input: MuxSchemeInput) -> bool:
    return input is None or input == "B" or isinstance(input, MuxSchemeBufferSpace)
