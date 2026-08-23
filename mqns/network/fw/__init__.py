from mqns.network.fw.controller import RoutingController
from mqns.network.fw.cutoff import CutoffScheme, CutoffSchemeWaitTime, CutoffSchemeWaitTimeCounters
from mqns.network.fw.fib import Fib, FibPath, FibRequest
from mqns.network.fw.forwarder import Forwarder, ForwarderCounters, ForwarderInitKwargs
from mqns.network.fw.fw_module import ForwarderModule, fw_control_cmd_handler, fw_signaling_cmd_handler
from mqns.network.fw.fw_nb import ForwarderNorthbound
from mqns.network.fw.message import MultiplexingVector, QubitKeySequence, SwapSequence
from mqns.network.fw.mux import MuxScheme
from mqns.network.fw.routing import (
    MultiplexingVectorInput,
    RoutingPath,
    RoutingPathInitArgs,
    RoutingPathMulti,
    RoutingPathSingle,
    RoutingPathStatic,
)
from mqns.network.fw.select import MemoryEprIterator, MemoryEprTuple
from mqns.network.fw.swap_sequence import SwapPolicy, SwapSequenceInput, parse_swap_sequence

__all__ = [
    "CutoffScheme",
    "CutoffSchemeWaitTime",
    "CutoffSchemeWaitTimeCounters",
    "Fib",
    "FibPath",
    "FibRequest",
    "Forwarder",
    "ForwarderCounters",
    "ForwarderInitKwargs",
    "ForwarderModule",
    "ForwarderNorthbound",
    "fw_control_cmd_handler",
    "fw_signaling_cmd_handler",
    "MemoryEprIterator",
    "MemoryEprTuple",
    "MultiplexingVector",
    "MultiplexingVectorInput",
    "MuxScheme",
    "parse_swap_sequence",
    "QubitKeySequence",
    "RoutingController",
    "RoutingPath",
    "RoutingPathInitArgs",
    "RoutingPathMulti",
    "RoutingPathSingle",
    "RoutingPathStatic",
    "SwapPolicy",
    "SwapSequence",
    "SwapSequenceInput",
]

for name in __all__:
    if name in (
        "MemoryEprIterator",
        "MemoryEprTuple",
        "MultiplexingVector",
        "MultiplexingVectorInput",
        "QubitKeySequence",
        "SwapPolicy",
        "SwapSequence",
        "SwapSequenceInput",
    ):
        continue
    globals()[name].__module__ = __name__
