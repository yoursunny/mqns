from mqns.models.error.coherent import CoherentErrorModel
from mqns.models.error.dissipation import DissipationErrorModel
from mqns.models.error.error import ErrorModel, PerfectErrorModel
from mqns.models.error.input import ErrorModelInput, parse_error
from mqns.models.error.pauli import BitFlipErrorModel, DephaseErrorModel, DepolarErrorModel, PauliErrorModel
from mqns.models.error.time_decay import TimeDecayFunc, make_time_decay_func, time_decay_nop

__all__ = [
    "BitFlipErrorModel",
    "CoherentErrorModel",
    "DephaseErrorModel",
    "DepolarErrorModel",
    "DissipationErrorModel",
    "ErrorModel",
    "ErrorModelInput",
    "make_time_decay_func",
    "parse_error",
    "PauliErrorModel",
    "PerfectErrorModel",
    "time_decay_nop",
    "TimeDecayFunc",
]

for name in __all__:
    if name in ("ErrorModelInput", "TimeDecayFunc"):
        continue
    globals()[name].__module__ = __name__
