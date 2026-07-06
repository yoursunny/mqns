from typing import TYPE_CHECKING, Self, TypedDict, Unpack, final, overload, override

from mqns.entity.node import Controller, NodePair, split_node_pair
from mqns.simulator import Event, Time

if TYPE_CHECKING:
    from mqns.network.fw.routing import RoutingPath, RoutingPathInitArgs


class RequestInitArgs(TypedDict, total=False):
    """
    Request attributes.
    """

    active_period: tuple[Time, Time]
    """
    Time period in which the request is active / valid.
    Use ``Time.SENTINEL`` to indicate no restriction on either side.

    In Proactive-Centralized mode, the ``ProactiveRoutingController`` installs routing path(s)
    upon entering the active period and uninstalls them upon leaving the active period.
    The (un)installation takes effect in the data plane after a one-way delay
    on the classical control network, which is typically ``CTRL_DELAY``.

    In Reactive-Centralized mode, the ``ReactiveRoutingController`` only attempts to fulfill
    a request if it is within the active period, and ignores inactive requests.

    This attribute is not supported for any modes other than described above.
    """


class Request:
    """Requests entanglement pairs between a source and a destination."""

    def __init__(self, np: NodePair, /, **kwargs: Unpack[RequestInitArgs]):
        self.src, self.dst = split_node_pair(np)
        self.not_before, self.not_after = kwargs.get("active_period", (Time.SENTINEL, Time.SENTINEL))
        self.rp: "RoutingPath|None" = None
        self.rp_args: "RoutingPathInitArgs" = {}

    @overload
    def path(self, rp: "RoutingPath", /) -> Self:
        """
        Specify the routing path.

        Args:
            rp: A ``RoutingPath`` instance, whose src-dst pair must match this request.

        The routing path will be inserted to the centralized controller at specified times.
        If the network does not have a centralized controller, this has no effect.

        If both this overload and the ``RoutingPathInitArgs`` overload are called,
        the ``RoutingPath`` instance provided in this call takes priority.
        """

    @overload
    def path(self, /, **kwargs: Unpack["RoutingPathInitArgs"]) -> Self:
        """
        Update routing path constructor parameters.

        The routing path will be inserted to the centralized controller at specified times.
        If the network does not have a centralized controller, this has no effect.

        ``RoutingPath`` subclass is chosen based on network configuration:

        * If the network uses Yen routing algorithm, ``RoutingPathMulti``.
        * If the network uses buffer-space multiplexing scheme, ``RoutingPathSingle(QubitAllocationType.FOLLOW_QCHANNEL)``.
        * Otherwise, ``RoutingPathSingle(QubitAllocationType.DISABLED)``.
        """

    def path(self, rp: "RoutingPath|None" = None, /, **kwargs: Unpack["RoutingPathInitArgs"]) -> Self:
        if rp:
            assert rp.src == self.src
            assert rp.dst == self.dst
            self.rp = rp
        else:
            self.rp_args.update(kwargs)
        return self

    def in_active_period(self, t: Time) -> bool:
        """
        Determine if time point ``t`` is within active period.
        """
        if self.not_before is not Time.SENTINEL and t < self.not_before:
            return False
        if self.not_after is not Time.SENTINEL and t > self.not_after:
            return False
        return True

    @property
    def req_id(self) -> int:
        """Return ``req_id`` attribute, defaults to ``-1``."""
        return self.rp.req_id if self.rp else self.rp_args.get("req_id", -1)

    def __repr__(self) -> str:
        return f"<Request {self.src}-{self.dst}>"


@final
class RequestActiveEvent(Event):
    """Event when a request enters or exits active_period."""

    def __init__(self, node: Controller, req: Request, enter: bool, *, t: Time):
        super().__init__(t, f"RequestActiveEvent({req}, {enter})")
        self.node = node
        self.req = req
        self.enter = enter

    @override
    def invoke(self) -> None:
        self.node.handle(self)
