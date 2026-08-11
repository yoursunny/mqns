import math
from collections.abc import Iterable
from typing import Literal, NamedTuple, override

from tap import Tap

from mqns.network.builder import EprTypeLiteral, NetworkBuilder, tap_configure
from mqns.network.fw import Forwarder, ForwarderCounters
from mqns.network.protocol.classicbridge import ClassicBridge
from mqns.network.protocol.consumer import RequestCounters
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("INFO")


class Args(Tap):
    nats_prefix: str = ClassicBridge.DEFAULT_NATS_PREFIX  # prefix of NATS subjects
    sim_accuracy: int = 1_000_000  # simulation accuracy in time slots per second
    seed: int | None = None  # random seed
    mode: Literal["PCA", "RCS"] = "PCA"
    sync_timing: list[float]
    L: tuple[float, float] = (50, 10)  # qchannel lengths (km)
    M: tuple[int, int] = (2, 4)  # qchannel capacity
    t_cohere: float = 0.05  # memory coherence time (s)
    epr_type: EprTypeLiteral  # network-wide EPR type

    @override
    def configure(self) -> None:
        tap_configure(self)
        self.add_argument("--L", metavar=("L_edge", "L_center"))
        self.add_argument("--M", metavar=("M_edge", "M_center"))


class Stats(NamedTuple):
    path1: RequestCounters
    path2: RequestCounters
    fw: dict[str, ForwarderCounters]

    def _describe(self) -> Iterable[str]:
        yield f"PATH1: {self.path1}"
        yield f"PATH2: {self.path2}"
        yield "fw:"
        for node_name in "S1", "D1", "S2", "D2", "R1", "R2":
            yield f"  {node_name}: {self.fw[node_name]}"

    def __repr__(self) -> str:
        return "\n".join(self._describe())


def run_simulation(args: Args) -> Stats:
    rng.reseed(args.seed)

    b = NetworkBuilder(
        epr_type=args.epr_type,
    )
    b.topo(
        channels=[
            ("S1-R1", args.L[0], args.M[0]),
            ("S2-R1", args.L[0], args.M[0]),
            ("R1-R2", args.L[1], args.M[1]),
            ("R2-D1", args.L[0], args.M[0]),
            ("R2-D2", args.L[0], args.M[0]),
        ],
        t_cohere=args.t_cohere,
    )

    match args.mode:
        case "PCA":
            b.proactive_centralized()
        case "RCS":
            b.reactive_centralized(timing=args.sync_timing)

    b.external_controller(nats_prefix=args.nats_prefix)

    net = b.make_network()
    del b

    s = Simulator(0, math.inf, accuracy=args.sim_accuracy, install_to=(log, net))
    s.run()

    stats = Stats(
        path1=RequestCounters.of(net, 10, "S1-D1"),
        path2=RequestCounters.of(net, 20, "S2-D2"),
        fw={},
    )
    for node in net.nodes:
        stats.fw[node.name] = node.get_app(Forwarder).cnt
    return stats


if __name__ == "__main__":
    args = Args().parse_args()
    stats = run_simulation(args)
    print("")
    print("---- STATS ----")
    print(stats)
