from typing import Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


MultiplexingMode = Literal["B", "S"]
MultiplexingVector = list[tuple[int, int]]


class PathInstructions(TypedDict):
    req_id: int
    route: list[str]
    swap: list[int]
    mux: MultiplexingMode
    m_v: NotRequired[MultiplexingVector]
    purif: dict[str, int]


class InstallPathMsg(TypedDict):
    cmd: Literal["install_path"]
    path_id: int
    instructions: PathInstructions


class PurifMsgBase(TypedDict):
    path_id: int
    purif_node: str
    partner: str
    epr: str
    measure_epr: str
    round: int


class PurifSolicitMsg(PurifMsgBase):
    cmd: Literal["PURIF_SOLICIT"]


class PurifResponseMsg(PurifMsgBase):
    cmd: Literal["PURIF_RESPONSE"]
    result: bool


class SwapUpdateMsg(TypedDict):
    cmd: Literal["SWAP_UPDATE"]
    path_id: int
    swapping_node: str
    partner: str
    epr: str
    new_epr: str | None  # None means swapping failed
