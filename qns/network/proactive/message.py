from typing import Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


MultiplexingVector = list[tuple[int, int]]


class PathInstructions(TypedDict):
    req_id: int
    route: list[str]
    swap: list[int]
    m_v: NotRequired[MultiplexingVector]
    purif: dict[str, int]


def validate_path_instructions(instructions: PathInstructions) -> None:
    def check_purif_segment(segment_name: str) -> bool:
        try:
            idx0, idx1 = [route.index(node_name) for node_name in segment_name.split("-")]
            return idx0 < idx1
        except ValueError:
            return False

    route = instructions["route"]
    if len(route) != len(instructions["swap"]) or len(route) == 0:
        raise ValueError("swapping order does not match route length")

    if "m_v" in instructions and len(instructions["m_v"]) != len(route) - 1:
        raise ValueError("multiplexing vector does not match route length")

    for segment_name in instructions["purif"].keys():
        if not check_purif_segment(segment_name):
            raise ValueError(f"purif segment {segment_name} does not exist in route")


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
