from collections.abc import Mapping
from typing import Literal

from mqns.models.epr.entanglement import Entanglement
from mqns.models.epr.mixed import MixedStateEntanglement
from mqns.models.epr.werner import WernerStateEntanglement

type EprTypeLiteral = Literal["W", "M"]
"""
String representation of commonly used entanglement models.
"""

EPR_TYPE_MAP: Mapping[EprTypeLiteral, type[Entanglement]] = {
    "W": WernerStateEntanglement,
    "M": MixedStateEntanglement,
}

type EprTypeInput = type[Entanglement] | EprTypeLiteral | None
"""
Entanglement model input parsable by ``parse_epr_type``.

* ``None`` or ``"W"``: Werner state.
* ``"M"``: Bell-diagonal state.
* ``Entanglement`` subclass constructor: use the given type.
"""


def parse_epr_type(input: EprTypeInput) -> type[Entanglement]:
    """
    Parse an entanglement model input.
    """
    if input is None:
        return WernerStateEntanglement

    if isinstance(input, str):
        return EPR_TYPE_MAP[input]

    return input
