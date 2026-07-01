from collections import defaultdict
from typing import Protocol, override

import numpy as np

from mqns.entity.memory import QubitState
from mqns.entity.node import Application, QNode
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.event import EntanglementReadyEvent, QubitReleasedEvent
from mqns.simulator import event_handler
from mqns.utils import json_encodable, log


class RequestLike(Protocol):
    req_id: int
    src: str
    dst: str


@json_encodable
class RequestCounters:
    """
    Consumption counters for an end-to-end request.

    Each entangled pair is delivered to two consumers, possibly at different times due to heralding timing.
    However, only the consumer that measures the second qubit logs the consumption,
    because the final end-to-end fidelity can only be calculated after applying memory error models on both sides.
    """

    n_consumed = 0
    """How many entanglements were consumed (either end-to-end or in swap-disabled mode)."""
    consumed_sum_fidelity = 0.0
    """
    Sum of fidelity of consumed entanglements.
    """
    consumed_fidelity_values: list[float] | None = None
    """
    Fidelity values of consumed entanglements, None disables collection.
    """

    @staticmethod
    def of(net: QuantumNetwork, req: RequestLike) -> "RequestCounters":
        """
        Obtain consumption counters of a request.

        Args:
            net: Quantum network.
            req: ``RoutingPath`` instance.

        Returns: ConsumerPathCounters for the request, aggregated from src and dst nodes.
        """
        a = net.get_node(req.src).get_app(Consumer).cnt.get(req.req_id, RequestCounters())
        b = net.get_node(req.dst).get_app(Consumer).cnt.get(req.req_id, RequestCounters())

        g = RequestCounters()
        g.n_consumed = a.n_consumed + b.n_consumed
        g.consumed_sum_fidelity = a.consumed_sum_fidelity + b.consumed_sum_fidelity
        if a.consumed_fidelity_values is b.consumed_fidelity_values:
            g.consumed_fidelity_values = a.consumed_fidelity_values
        return g

    @staticmethod
    def enable_collect_all(net: QuantumNetwork, req: RequestLike) -> None:
        """
        Enable collecting all values for histogram generation.

        Args:
            net: Quantum network.
            req: ``RoutingPath`` instance.
        """
        a = net.get_node(req.src).get_app(Consumer).cnt[req.req_id]
        b = net.get_node(req.dst).get_app(Consumer).cnt[req.req_id]

        assert a.consumed_fidelity_values is None
        assert b.consumed_fidelity_values is None
        a.consumed_fidelity_values = b.consumed_fidelity_values = []

    def increment_n_consumed(self, fidelity: float) -> None:
        self.n_consumed += 1
        self.consumed_sum_fidelity += fidelity
        if self.consumed_fidelity_values is not None:
            self.consumed_fidelity_values.append(fidelity)

    @property
    def consumed_avg_fidelity(self) -> float:
        """Average fidelity of consumed entanglements."""
        if self.consumed_fidelity_values is not None and len(self.consumed_fidelity_values) == self.n_consumed > 0:
            return np.mean(self.consumed_fidelity_values).item()
        return self.get_per_consumed(self.consumed_sum_fidelity)

    def get_rate(self, duration: float) -> float:
        """
        Calculate entanglement rate.

        Args:
            duration: How many seconds did the path remain active.

        Returns: Entanglement rate in entanglements per second.
        """
        return self.n_consumed / duration

    def get_per_consumed(self, x: float) -> float:
        """
        Divide a value by ``n_consumed``, but return zero if ``n_consumed`` is zero.
        """
        return x / self.n_consumed if self.n_consumed > 0 else 0.0

    def __repr__(self) -> str:
        return f"consumed={self.n_consumed} (F={self.consumed_avg_fidelity})"


class Consumer(Application[QNode]):
    """
    T-class application that consumes entanglements.
    """

    def __init__(self):
        self.cnt = defaultdict[int, RequestCounters](RequestCounters)
        """
        Path counters keyed by req_id.
        """

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""

    @event_handler
    def handle_ready(self, event: EntanglementReadyEvent) -> None:
        qubit = event.qubit
        epr = event.epr
        req_id = event.req_id

        role_str = "first"
        if epr.consume_with_store_decay_side(self.simulator.tc, side=0 if epr.src is self.node else 1):
            role_str = "second"
            self.cnt[req_id].increment_n_consumed(epr.fidelity)
        log.debug(f"{self}: consume EPR {role_str} for request {req_id}: {epr}")

        self.memory.read(qubit.addr, remove=True)
        qubit.state = QubitState.RELEASE
        self.simulator.add_event(QubitReleasedEvent(self.node, qubit, t=self.simulator.tc))
