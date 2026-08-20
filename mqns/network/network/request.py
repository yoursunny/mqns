from typing import TYPE_CHECKING, Final, Self, TypedDict, Unpack, cast, final, overload, override

from mqns.entity.node import Controller, NodePair, split_node_pair
from mqns.simulator import Event, EventHandleSlot, Time
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.routing import RoutingPath, RoutingPathInitArgs


class RequestInitArgs(TypedDict, total=False):
    """
    Request attributes.
    """

    active_period: tuple[Time | float, Time | float]
    """
    Time period in which the request is active / valid.
    Use ``Time.MIN`` and ``Time.MAX`` to indicate no restriction.

    In Proactive-Centralized mode, the ``ProactiveRoutingController`` installs routing path(s)
    upon entering the active period and uninstalls them upon leaving the active period.
    The (un)installation takes effect in the data plane after a one-way delay
    on the classical control network, which is typically ``CTRL_DELAY``.

    In Reactive-Centralized mode, the ``ReactiveRoutingController`` only attempts to fulfill
    a request if it is within the active period, and ignores inactive requests.

    This attribute is not supported for any modes other than described above.
    """

    epr_count: int
    """
    How many entangled pairs are desired.
    Use ``-1`` or omit this attribute to indicate no limitation.

    If this quantity of EPRs have been delivered to end-node applications, the network considers
    this request fulfilled and may release its resources.

    In Proactive-Centralized mode, the ``ProactiveRoutingController`` sends ``ConsumeCountBeginMsg``
    to both end-nodes, the ``ProactiveForwarder`` at each end-node replies with ``ConsumeCountEndMsg``.
    After receiving both replies, the controller uninstalls the path(s).

    This attribute is not supported for any modes other than described above.
    """


class Request:
    """Requests entanglement pairs between a source and a destination."""

    src: Final[str]
    """Source node name."""
    dst: Final[str]
    """Destination node name."""

    active_since: Time | float
    """
    Active period lower bound (inclusive), ``Time.MIN`` means no restriction.
    This field is guaranteed to be ``Time`` after the request is added to a network and a simulator is installed.
    """
    active_until: Time | float
    """
    Active period upper bound (exclusive), ``Time.MAX`` means no restriction.
    This field is guaranteed to be ``Time`` after the request is added to a network and a simulator is installed.
    """
    inactive_event: EventHandleSlot["RequestInactiveEvent"]
    """
    Event when the request becomes inactive.
    This field is assigned by the controller.
    """

    epr_count: Final[int]
    """
    How many entangled pairs are desired, ``-1`` means no restriction.
    """
    epr_count_await: set[str] = {"#epr_count_disabled"}  # see .is_active() for explanation of class-level default
    """
    Which end nodes have not reported reaching ``epr_count``.
    """

    rp: "RoutingPath|None" = None
    """Routing path specified by scenario or assigned by controller."""
    rp_args: "RoutingPathInitArgs"
    """Routing path parameters specified by scenario and used by controller."""

    @overload
    def __init__(self, np: NodePair, /, **kwargs: Unpack[RequestInitArgs]):
        """
        Construct from src-dst node pair.
        """

    @overload
    def __init__(self, rp: "RoutingPath", /, **kwargs: Unpack[RequestInitArgs]):
        """
        Construct from ``RoutingPath``, equivalent to calling ``.path(rp)``.
        """

    def __init__(self, arg1: "NodePair|RoutingPath", /, **kwargs: Unpack[RequestInitArgs]):
        self.inactive_event = EventHandleSlot()
        if isinstance(arg1, str | tuple):
            self.src, self.dst = split_node_pair(cast(NodePair, arg1))
        else:
            self.rp = arg1
            self.src, self.dst = self.rp.src, self.rp.dst
        self.active_since, self.active_until = kwargs.get("active_period", (Time.MIN, Time.MAX))
        self.epr_count = kwargs.get("epr_count", -1)
        if self.epr_count > 0:
            self.epr_count_await = {self.src, self.dst}
        else:
            assert self.epr_count == -1
        self.rp_args = {}

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
        * Otherwise, ``RoutingPathSingle``.
        """

    def path(self, rp: "RoutingPath|None" = None, /, **kwargs: Unpack["RoutingPathInitArgs"]) -> Self:
        if rp:
            assert rp.src == self.src
            assert rp.dst == self.dst
            self.rp = rp
        else:
            self.rp_args.update(kwargs)
        return self

    def is_active(self, t: Time) -> bool:
        """
        Determine if the request is active, subject to these conditions:

        * ``epr_count`` has not been reached.
        * Time point ``t`` is within active period.
        """
        if not self.epr_count_await:
            # If .epr_count is enabled, .epr_count_await becomes empty when both end-nodes reports reaching the count.
            # If .epr_count is disabled, class-level default is a non-empty set so that `not .epr_count_await` remains False.
            return False
        if t < cast(Time, self.active_since):
            return False
        if t >= cast(Time, self.active_until):
            return False
        return True

    @property
    def req_id(self) -> int:
        """Return ``req_id`` attribute, defaults to ``-1``."""
        return self.rp.req_id if self.rp else self.rp_args.get("req_id", -1)

    def __repr__(self) -> str:
        tokens = [f"{self.src}-{self.dst}", f"active_period={self.active_since}-{self.active_until}"]
        if self.epr_count > 0:
            tokens.append(f"epr_count={self.epr_count}")
        return f"Request({', '.join(tokens)})"


@final
class RequestActiveEvent(Event):
    """Event when a request becomes active."""

    def __init__(self, node: Controller | None, req: Request, *, t: Time):
        super().__init__(t, f"{req}")
        self.node = node
        self.req = req

    @override
    def invoke(self) -> None:
        log.info("NETWORK: REQ_ACTIVE %s", self.req)
        if self.node:
            self.node.handle(self)


@final
class RequestInactiveEvent(Event):
    """Event when a request becomes inactive."""

    def __init__(self, node: Controller | None, req: Request, *, t: Time):
        super().__init__(t, f"{req}")
        self.node = node
        self.req = req

    @override
    def invoke(self) -> None:
        log.info("NETWORK: REQ_INACTIVE %s", self.req)
        if self.node:
            self.node.handle(self)
