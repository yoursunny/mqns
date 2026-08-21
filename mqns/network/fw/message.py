from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

type MultiplexingVector = Sequence[int]
"""Multiplexing vector -- guides memory allocation in buffer-space multiplexing scheme."""

type QubitKeySequence = Sequence[str]
"""Qubit key sequence -- identifies specific qubits in reactive forwarding."""

type SwapSequence = Sequence[int]
"""Swap sequence -- nonnegative integers to control swapping order."""


class PathInstructions(TypedDict):
    """
    Swapping and purification instructions for the forwarders.
    """

    path_id: int
    """
    Path identifier -- nonnegative integer to identify this path.
    """

    route: list[str]
    """
    Path vector -- a list of node names, in the order they appear in the path.

    There must a quantum channel and a classical channel between each pair of adjacent nodes.
    """

    bufferspace_mv: NotRequired[MultiplexingVector]
    """
    Multiplexing vector, used by ``proactive.MuxSchemeBufferSpace``.

    This list shall have two integers per qchannel, i.e. ``2*(len(route)-1)`` length.
    The (2i)-th and (2i+1)-th integer corresponds to left and right node of the i-th qchannel.
    Each integer indicates how many qubits should the node allocate to this path on that qchannel,
    among the memory qubits assigned to that qchannel by the node.
    ``0`` means allocating all qubits assigned to that qchannel to this path.

    Example:

        route          = [S,    R,    D]
        bufferspace_mv = [  4,2,  3,0  ]

    * S should allocate 4 qubits on S-R channel.
    * R should allocate 2 qubits on S-R channel and 3 qubits on R-D channel.
    * D should allocate all qubits assigned to R-D channel.
    """

    reactive_qubits: NotRequired[QubitKeySequence]
    """
    Qubit key sequence, used in reactive forwarding.

    This list shall have one string per qchannel, i.e. ``len(route)-1`` length.
    Each string is a qubit reservation key that should be swapped into the end-to-end entanglement.
    """

    swap: SwapSequence
    """
    Swap sequence -- nonnegative integers to control swapping order.

    This list shall have the same length as ``route``.
    Each element represents swapping rank of the corresponding node.
    A node with smaller rank shall perform swapping before a node with larger rank.

    To disable swapping, set this to a list of zeros.
    When swapping is disabled, the forwarder will consume entanglement upon completing purification,
    without attempting entanglement swapping.
    """

    swap_cutoff: NotRequired[list[int]]
    """
    Swap cutoff time -- maximum age at each swapping step.

    This list shall have two elements per intermediate node, i.e. ``2*(len(route)-2)`` length.
    The (2i)-th and (2i+1)-th element corresponds to left and right qchannel of the i-th intermediate node.
    Each element is a duration in time slots (see ``Time`` class); ``-1`` means no restriction.

    The semantics of "age" depend on the CutoffScheme passed to Forwarder.
    If swapping has been disabled, this list has no effect.
    If this list is omitted, it is equivalent to all ``-1`` i.e. no restriction on any node.
    """

    purif: dict[str, int]
    """
    Purification scheme.

    Each key is a segment name consists of two node names concatenated with a hyphen (``-``),
    where the nodes appear in the same order as in the route but do not have to be adjacent.
    Each value is an integer of the required rounds of purification at this segment.
    The default for every segment is zero i.e. no purification is performed.
    """


def _check_purif_segment(segment_name: str, route: Sequence[str]) -> bool:
    try:
        idx0, idx1 = (route.index(node_name) for node_name in segment_name.split("-"))
        return idx0 < idx1
    except ValueError:
        return False


def validate_path_instructions(
    inst: PathInstructions,
    *,
    bufferspace: bool | None = False,
    reactive: bool | None = False,
) -> None:
    """
    Validate ``PathInstructions``.

    Args:
        inst: PathInstructions to be validated.
        bufferspace: Require the presence or absence of ``bufferspace_*`` fields.
        reactive: Require the presence or absence of ``reactive_*`` fields.
    """
    if inst["path_id"] < 0:
        raise ValueError("path_id is negative")

    route = inst["route"]
    n = len(route)
    if n == 0:
        raise ValueError("route is empty")

    if bufferspace not in (None, "bufferspace_mv" in inst):
        raise ValueError(f"bufferspace_mv must be {'present' if bufferspace else 'absent'}")
    if "bufferspace_mv" in inst and len(inst["bufferspace_mv"]) != 2 * (n - 1):
        raise ValueError("bufferspace_mv does not match route length")

    if reactive not in (None, "reactive_qubits" in inst):
        raise ValueError(f"reactive_qubits must be {'present' if reactive else 'absent'}")
    if "reactive_qubits" in inst and len(inst["reactive_qubits"]) != n - 1:
        raise ValueError("reactive_qubits does not match route length")

    if len(inst["swap"]) != len(route):
        raise ValueError("swapping order does not match route length")

    if "swap_cutoff" in inst and len(inst["swap_cutoff"]) != 2 * (n - 2):
        raise ValueError("swap_cutoff does not match route length")

    for segment_name in inst["purif"].keys():
        if not _check_purif_segment(segment_name, route):
            raise ValueError(f"purif segment {segment_name} does not exist in route")


class PathInsertMsg(TypedDict):
    """
    Path insertion command from controller to forwarders.

    In Proactive-Centralized mode, this command installs one or more routing paths,
    which is persisted until they are deleted via ``PathDeleteMsg``.

    In Reactive-Centralized mode, this command gives specific swapping instructions,
    which is valid for one time slot and automatically deleted after the slot.

    This command is not allowed for any modes other than described above.
    """

    cmd: Literal["PATH_INSERT"]
    req_id: int
    """
    Request identifier -- nonnegative integer to identify a request between a src-dst pair.

    Each routing path belongs to a request, which identifies the src-dst pair along with other attributes.
    All routing paths that belong to the same request must be sent in the same ``PathInsertMsg``.
    However, it is possible for multiple requests to have the same src-dst pair.
    """

    epr_count: NotRequired[int]
    """
    How many entangled pairs are desired.

    If this is nonnegative, each end-node forwarder must send ``PathReachEprCountMsg`` when
    sufficient quantity of entangled pairs have been delivered for consumption.
    """

    paths: list[PathInstructions]
    """
    Routing paths, nonempty list.
    """


class PathDeleteMsg(TypedDict):
    """
    Path deletion command from controller to forwarders.

    In Proactive-Centralized mode, this command deletes the routing paths
    installed with ``PathInsertMsg``.

    This command is not allowed for any modes other than described above.
    """

    cmd: Literal["PATH_DELETE"]
    req_id: int
    """
    Request identifier -- nonnegative integer to identify a request between a src-dst pair.
    """


class PathReachEprCountMsg(TypedDict):
    """
    Reaching ``epr_count`` report message from end-node forwarder to controller.
    """

    cmd: Literal["PATH_REACH_EPR_COUNT"]
    req_id: int
    """
    Request identifier -- nonnegative integer to identify a request between a src-dst pair.
    """


class CutoffDiscardMsg(TypedDict):
    cmd: Literal["CUTOFF_DISCARD"]
    path_id: int
    key: str
    round: int


class PurifMsgBase(TypedDict):
    path_id: int
    purif_node: str
    partner: str
    key0: str
    key1: str
    round: int


class PurifSolicitMsg(PurifMsgBase):
    cmd: Literal["PURIF_SOLICIT"]


class PurifResponseMsg(PurifMsgBase):
    cmd: Literal["PURIF_RESPONSE"]
    result: bool


class SwapUpdateMsg(TypedDict):
    """Heralding message after swapping."""

    cmd: Literal["SWAP_UPDATE"]
    path_id: int
    """FIB entry path ID to guide classical forwarding of this message."""

    swapper: str
    """
    Node that performed the swap.
    This would be the sender of this message.
    """

    ends: list[str]
    """
    Left and right ends of the swapped EPR.

    * [0] and [2] are the two node names, one of which would be the recipient of this message.
    * [1] and [3] are the corresponding qubit reservation keys.
    """

    expiry: int
    """
    If zero, indicates swapping failure.
    If positive, time slot of qubit decoherence based on heralded knowledge.
    """
    q_paths: list[int]
    """Possible path IDs for the entanglement between left and right ends."""
