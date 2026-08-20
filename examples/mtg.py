"""
Demonstrate ``MatrixTrafficGenerator`` usage.

1. Generate an empty traffic definition file:

    python mtg.py --tm_gen >mtg.tm.toml

2. Modify the traffic definition file.
   At least one traffic flow must have non-zero probability.

3. Run the simulation with traffic definition file.

    python mtg.py --tm mtg.tm.toml --csv mtg.csv --sim_duration 10 --rate 10

CSV output includes per-request statistics:

* Request attributes: end-nodes, active period, requested EPR quantity.
* Consumed EPR quantity, average fidelity.
* Latency since request arrival (start of active period) and first/last EPR delivery.
  * This includes ``CTRL_DELAY`` in Proactive-Centralized mode.
"""

import csv
import itertools
import sys
import tomllib
from typing import Literal, TypedDict, cast, override

import tomli_w
from tap import Tap

from mqns.network.builder import NetworkBuilder
from mqns.network.fw import MuxSchemeBufferSpace, MuxSchemeStatistical
from mqns.network.network import MatrixTrafficGenerator, QuantumNetwork, Request, TrafficMatrixMapping
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator, Time
from mqns.utils import log, rng, seed_env

from examples_common.topo_multiplexing import END_NODES, define_topo

log.set_default_level("CRITICAL")


class Args(Tap):
    tm_gen: bool = False  # write empty traffic definition to stdout
    tm: str = ""  # network traffic definition file
    sim_duration: float = 1.0  # simulation duration in seconds
    mux: Literal["B", "S"] = "B"  # multiplexing scheme
    rate: float = 1.0  # request arrival rate (req/s)
    csv: str = ""  # save results as CSV file

    @override
    def process_args(self) -> None:
        if not self.tm_gen and not self.tm:
            raise ValueError("--tm is required if --tm_gen is absent")


class NetTrafficDef(TypedDict):
    matrix: dict[str, float]
    """
    Traffic matrix.
    Key: source and destination node, delimited by hyphen.
    Value: relative weight.
    """
    duration: float
    """
    Active duration per request, in seconds.
    """
    epr_count: int
    """
    Desired quantity of entangled pairs per request.
    ``-1`` means infinite.
    """


def gen_tm() -> None:
    d = NetTrafficDef(matrix={}, duration=1, epr_count=-1)
    for src, dst in itertools.product(END_NODES, END_NODES):
        if src == dst:
            continue
        d["matrix"][f"{src}-{dst}"] = 0.0
    tomli_w.dump(d, sys.stdout.buffer)


def build_network(args: Args) -> QuantumNetwork:
    b = NetworkBuilder()
    define_topo(b)
    match args.mux:
        case "B":
            b.proactive_centralized(mux=MuxSchemeBufferSpace(), mv_auto=1)
        case "S":
            b.proactive_centralized(mux=MuxSchemeStatistical())
    return b.make_network()


type Result = list[tuple[Request, RequestCounters]]


def run_simulation(seed: int, args: Args, tm: NetTrafficDef) -> Result:
    rng.reseed(seed)
    net = build_network(args)

    mtg = MatrixTrafficGenerator(
        net,
        cast(TrafficMatrixMapping, tm["matrix"]),
        sched="eager",
        rate=args.rate,
        duration=tm["duration"],
        epr_count=tm["epr_count"],
    )

    s = Simulator(0, args.sim_duration, accuracy=1000000, install_to=(log, net, mtg))
    s.run()

    return [(req, RequestCounters.of(net, req)) for req in net.requests]


def save_csv(args: Args, result: Result) -> None:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            (
                "req_id",
                "src",
                "dst",
                "active_since",
                "active_until",
                "req_epr_count",
                "n_consumed",
                "fid",
                "lat_first",
                "lat_last",
            )
        )
        for req, cnt in result:
            active_since = cast(Time, req.active_since)
            active_until = cast(Time, req.active_until)
            lat_first, lat_last = cnt.get_latency(active_since)
            w.writerow(
                (
                    req.req_id,
                    req.src,
                    req.dst,
                    active_since,
                    active_until,
                    req.epr_count,
                    cnt.n_consumed,
                    cnt.consumed_avg_fidelity,
                    lat_first,
                    lat_last,
                )
            )


def main(args: Args) -> None:
    if args.tm_gen:
        return gen_tm()

    with open(args.tm, "rb") as f:
        tm = cast(NetTrafficDef, tomllib.load(f))

    result = run_simulation(seed_env(0), args, tm)
    if args.csv:
        save_csv(args, result)

    total_consumed = sum(cnt.n_consumed for _, cnt in result)
    print(f"Simulation completed, {len(result)} requests, {total_consumed} consumed")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
