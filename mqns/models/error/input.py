import copy
from collections.abc import Callable
from typing import Never, NotRequired, TypedDict, cast

from mqns.models.error.dissipation import DissipationErrorModel
from mqns.models.error.error import ErrorModel, PerfectErrorModel
from mqns.models.error.pauli import DephaseErrorModel, DepolarErrorModel

type ErrorModelConstructor = Callable[[], ErrorModel]


class ErrorModelDictPSurvival(TypedDict):
    p_survival: float


class ErrorModelDictPError(TypedDict):
    p_error: float


class ErrorModelDictTime(TypedDict):
    t: NotRequired[float]
    rate: float


class ErrorModelDictLength(TypedDict):
    length: NotRequired[float]
    rate: float


def _apply_input_dict(
    error: ErrorModel,
    d: ErrorModelDictPSurvival | ErrorModelDictPError | ErrorModelDictTime | ErrorModelDictLength,
    dflt_t: float,
):
    if (p_survival := cast(ErrorModelDictPSurvival, d).get("p_survival")) is not None:
        return error.set(p_survival=p_survival)

    if (p_error := cast(ErrorModelDictPError, d).get("p_error")) is not None:
        return error.set(p_error=p_error)

    if (rate := cast(ErrorModelDictTime | ErrorModelDictLength, d).get("rate")) is not None:
        t = cast(ErrorModelDictTime, d).get("t", cast(ErrorModelDictLength, d).get("length", dflt_t))
        return error.set(t=t, rate=rate)

    return error


def _parse_error_str(input: str, dflt_t: float) -> ErrorModel:
    tokens = input.split(":")
    match tokens[0]:
        case "PERFECT":
            return PerfectErrorModel()
        case "DEPOLAR":
            m = DepolarErrorModel()
        case "DEPHASE":
            m = DephaseErrorModel()
        case "DISSIPATION":
            m = DissipationErrorModel()
        case _:
            raise ValueError("unrecognized ErrorModelInput string")

    try:
        value = float(tokens[1])
    except (ValueError, IndexError):
        raise ValueError("unrecognized ErrorModelInput string")

    if dflt_t < 0:
        m.set(p_error=value)
    else:
        m.set(t=dflt_t, rate=value)
    return m


type ErrorModelInput[D: ErrorModelDictTime | ErrorModelDictLength] = (
    ErrorModel
    | tuple[ErrorModel | ErrorModelConstructor, ErrorModelDictPSurvival | ErrorModelDictPError | D]
    | ErrorModelDictPSurvival
    | ErrorModelDictPError
    | D
    | str
    | None
)

type ErrorModelInputBasic = ErrorModelInput[Never]
"""``parse_error`` input, accepting dict with probability."""
type ErrorModelInputTime = ErrorModelInput[ErrorModelDictTime]
"""``parse_error`` input, accepting dict with time-based decay."""
type ErrorModelInputLength = ErrorModelInput[ErrorModelDictLength]
"""``parse_error`` input, accepting dict with length-based decay."""


def parse_error(
    input: ErrorModelInputBasic | ErrorModelInputTime | ErrorModelInputLength, dflt: ErrorModelConstructor, dflt_t: float
) -> ErrorModel:
    """
    Parse error model input.

    Args:
        input: input parameter.
        dflt: default ``ErrorModel`` subclass type.
        dflt_t: default ``t`` or ``length`` parameter; -1 if time/length based decay is unsupported.

    The input parameter could be one of:

    * ``None``: ``PerfectErrorModel``.
    * ``ErrorModel`` instance: used as is.
    * Dict: used as probabilities with default error model type.
    * ``ErrorModel`` instance and dict: clone and assign probabilities.
    * ``ErrorModel`` type and dict: construct and assign probabilities.
    * String: parsed following rules below.

    The dict could be one of:

    * ``{"p_survival":float}``: set survival probability.
    * ``{"p_error":float}``: set error probability.
    * ``{"t":float,"rate":float}``: set time based decay; ``dflt_t`` is used if ``t`` is omitted.
    * ``{"length":float,"rate":float}``: set length based decay; ``dflt_t`` is used if ``length`` is omitted.

    The string could be one of:

    * ``"PERFECT"``: ``PerfectErrorModel``.
    * ``DEPOLAR:rate``, ``DEPHASE:rate``, ``DISSIPATION:rate``: construct with specified rate,
      available when parsing memory / qchannel errors with time/length based decay (``dflt_t>=0``).
    * ``DEPOLAR:p_error``, ``DEPHASE:p_error``, ``DISSIPATION:p_error``: construct with specified error probability,
      available when parsing BSA / operate / measure errors (``dflt_t<0``).
    """
    if input is None:
        return PerfectErrorModel()
    if isinstance(input, ErrorModel):
        return input
    if isinstance(input, str):
        return _parse_error_str(input, dflt_t)

    base, d = input if isinstance(input, tuple) else (dflt, input)
    error = copy.deepcopy(base) if isinstance(base, ErrorModel) else base()
    return _apply_input_dict(error, d, dflt_t)
