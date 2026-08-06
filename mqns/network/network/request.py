from typing import TYPE_CHECKING, Any, Final, Self, TypedDict, Unpack, final, overload, override

from mqns.entity.node import Controller, NodePair, split_node_pair
from mqns.simulator import Event, Time

if TYPE_CHECKING:
    from mqns.network.fw.routing import RoutingPath, RoutingPathInitArgs


class RequestInitArgs(TypedDict, total=False):
    """
    Request attributes.
    """

    active_period: tuple[Time | float, Time | float]
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

    src: Final[str]
    """Source node name."""
    dst: Final[str]
    """Destination node name."""

    active_since: Time
    """
    Active period lower bound (inclusive), ``Time.MIN`` means no restriction.
    This field becomes valid when the request is added to a network and a simulator is installed into the network.
    """
    active_until: Time
    """
    Active period upper bound (exclusive), ``Time.MAX`` means no restriction.
    This field becomes valid when the request is added to a network and a simulator is installed into the network.
    """

    inactive_event: "RequestInactiveEvent|None" = None
    """Event when the request becomes inactive."""

    rp: "RoutingPath|None" = None
    """Routing path specified by scenario or assigned by controller."""
    rp_args: "RoutingPathInitArgs"
    """Routing path parameters specified by scenario and used by controller."""

    def __init__(self, np: NodePair, /, **kwargs: Unpack[RequestInitArgs]):
        self.src, self.dst = split_node_pair(np)
        self.active_since_input, self.active_until_input = kwargs.get("active_period", (Time.MIN, Time.MAX))
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

        * Time point ``t`` is within active period.
        """
        if t < self.active_since:
            return False
        if t >= self.active_until:
            return False
        return True

    @property
    def req_id(self) -> int:
        """Return ``req_id`` attribute, defaults to ``-1``."""
        return self.rp.req_id if self.rp else self.rp_args.get("req_id", -1)

    def __repr__(self) -> str:
        return f"Request({self.src}-{self.dst}, active_period={self._repr_active('since')}-{self._repr_active('until')})"

    def _repr_active(self, key: str) -> Any:
        return getattr(self, f"active_{key}", None) or getattr(self, f"active_{key}_input", None)


@final
class RequestActiveEvent(Event):
    """Event when a request becomes active."""

    def __init__(self, node: Controller, req: Request, *, t: Time):
        super().__init__(t, f"{req}")
        self.node = node
        self.req = req

    @override
    def invoke(self) -> None:
        self.node.handle(self)


@final
class RequestInactiveEvent(Event):
    """Event when a request becomes inactive."""

    def __init__(self, node: Controller, req: Request, *, t: Time):
        super().__init__(t, f"{req}")
        self.node = node
        self.req = req

    @override
    def invoke(self) -> None:
        self.node.handle(self)
