from collections.abc import Iterator, Mapping
from typing import Literal, TypedDict, Unpack, cast

import numpy as np

from mqns.entity.node import NodePair, split_node_pair
from mqns.models.core import TrafficMatrix, TrafficMatrixInput
from mqns.network.network.network import QuantumNetwork
from mqns.network.network.request import Request
from mqns.simulator import Simulator, Time, func_to_event
from mqns.utils import rng

type TrafficMatrixMapping = Mapping[NodePair, float]
"""
Mapping input format for traffic matrix.
Each key is a src-dst node pair.
Each value is a nonnegative relative probability.
Missing node pairs have zero probability.
"""


class MatrixTrafficGeneratorInitArgs(TypedDict, total=False):
    sched: Literal["never", "eager", "lazy"]
    """
    How should ``MatrixTrafficGenerator.install()`` schedule the requests.

    * "never": Do not schedule requests automatically.
      Requests may be retrieved with ``.iter()`` method.
    * "eager": Schedule all requests upfront.
      This is incompatible with continuous simulation.
    * "lazy": Schedule the first request, then schedule the next request
      upon the previous request's arrival. This is the default.
    """

    rate: float
    """
    Request arrival rate in Hz, defaults to 1 Hz.
    This is expected number of requests per second.
    """

    duration: Time | float
    """
    Request active period duration, defaults to 1 second.
    """

    epr_count: int
    """
    Desired quantity of entangled pairs per request, defaults to 1.
    ``-1`` means infinity (subject to ``duration``); ``0`` is invalid.
    """


class MatrixTrafficGenerator:
    """
    Generate requests with Poisson process and traffic matrix.
    """

    def __init__(
        self,
        net: QuantumNetwork,
        tm: TrafficMatrixInput | TrafficMatrixMapping,
        **kwargs: Unpack[MatrixTrafficGeneratorInitArgs],
    ):
        """
        Args:
            net: The quantum network.
            tm: Traffic matrix input that matches the quantum network.
        """
        self.net = net
        self.node_names = [node.name for node in net.nodes]
        self.tm = TrafficMatrix(self._convert_tm_map(tm) if isinstance(tm, Mapping) else tm, len(net.nodes))

        self._sched = kwargs.get("sched", "lazy")
        self._scale = 1 / kwargs.get("rate", 1.0)
        self._duration = kwargs.get("duration", 1.0)
        self._epr_count = kwargs.get("epr_count", 1)

    def _convert_tm_map(self, tm: TrafficMatrixMapping):
        n = len(self.node_names)
        array = np.zeros((n, n), dtype=np.float64)
        for pair, prob in tm.items():
            src, dst = split_node_pair(pair)
            try:
                src_i = self.node_names.index(src)
                dst_i = self.node_names.index(dst)
            except ValueError:
                raise ValueError(f"node not found for node pair {src}-{dst}")
            array[src_i, dst_i] = prob
        return array

    def install(self, simulator: Simulator):
        """
        Install this traffic generator into a simulator.
        """
        self.simulator = simulator
        self._duration = Time.from_time_or_sec(self._duration, accuracy=simulator.accuracy)

        match self._sched:
            case "eager":
                self._sched_eager()
            case "lazy":
                self._sched_lazy(self.iter())

    def iter(self) -> Iterator[Request]:
        """
        Generates requests with Poisson process and traffic matrix.

        Returns: A (possibly infinite) iterator of requests.
        """
        now = self.simulator.ts.sec
        end = np.inf if self.simulator.te is Time.SENTINEL else self.simulator.te.sec
        while True:
            now += rng.exponential(scale=self._scale)
            if now > end:
                break
            t = self.simulator.time(sec=now)
            yield self._build_request(t)

    def _build_request(self, t: Time) -> Request:
        """
        Build request with specified arrival time.
        """
        src, dst = self.tm.sample()
        return Request(
            (self.node_names[src], self.node_names[dst]),
            active_period=(t, t + self._duration),
            epr_count=self._epr_count,
        )

    def _sched_eager(self):
        assert not self.simulator.is_continuous, "sched=eager is incompatible with continuous simulation"
        self.net.add_request(*self.iter())

    def _sched_lazy(self, it: Iterator[Request]):
        req = next(it, None)
        if not req:
            return

        self.net.add_request(req)
        self.simulator.sched(func_to_event(cast(Time, req.active_since), self._sched_lazy, it))
