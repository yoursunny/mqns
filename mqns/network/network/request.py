from typing import TypedDict, Unpack

from mqns.entity.node import QNode


class RequestAttr(TypedDict, total=False):
    """
    Request attributes.
    """

    req_id: int
    """
    Request identifier, identifies source-destination pair.
    Default is auto-generated.
    Specifying this value allows retrieving consumer counters.
    """


class Request:
    """Requests entanglement pairs between a source and a destination."""

    def __init__(self, src: QNode, dst: QNode, **attr: Unpack[RequestAttr]):
        """
        Args:
            src: Left node to receive one of the entangled qubits.
            dst: Right node to receive one of the entangled qubits.
            attr: Request attributes.
        """
        self.src = src
        self.dst = dst
        self.attr = attr

    @property
    def req_id(self) -> int:
        """Return ``req_id`` attribute, defaults to ``-1``."""
        return self.attr.get("req_id", -1)

    def __repr__(self) -> str:
        return f"<Request {self.src}-{self.dst}>"
