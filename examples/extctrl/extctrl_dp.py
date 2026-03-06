import math
from typing import Literal, override

from tap import Tap

from mqns.network.builder import EprTypeLiteral, NetworkBuilder, tap_configure
from mqns.simulator import Simulator
from mqns.utils import log, rng

log.set_default_level("CRITICAL")

SIMULATOR_ACCURACY = 1000000


class Args(Tap):
    seed: int = 0  # random seed
    mode: Literal["P", "R"] = "P"  # choose proactive or reactive mode
    epr_type: EprTypeLiteral  # network-wide EPR type

    @override
    def configure(self) -> None:
        super().configure()
        tap_configure(self)


def run_simulation(args: Args):
    rng.reseed(args.seed)

    b = NetworkBuilder(
        epr_type=args.epr_type,
    )
    b.topo(
        channels=[
            ("S1-R1", 10, 2),
            ("S2-R1", 10, 2),
            ("R1-R2", 10, 4),
            ("R2-D1", 10, 2),
            ("R2-D2", 10, 2),
        ]
    )

    match args.mode:
        case "P":
            b.proactive_centralized()
        case "R":
            b.reactive_centralized()

    b.external_controller()

    net = b.make_network()
    del b

    s = Simulator(0, math.inf, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()


if __name__ == "__main__":
    args = Args().parse_args()
    run_simulation(args)
