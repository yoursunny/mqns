import time
from typing import Any, TypedDict

from tap import Tap


# Command line arguments
class RunArgs(Tap):
    seed: int = -1  # random seed number
    nnodes: int = 16  # network size - number of nodes
    nedges: int = 20  # network size - number of edges
    sim_duration: float = 1.0  # simulation duration in seconds
    qchannel_capacity: int = 10  # quantum channel capacity
    time_limit: float = 10800.0  # wall-clock limit in seconds
    outdir: str = "."  # output directory


def parse_run_args() -> RunArgs:
    args = RunArgs().parse_args()
    if args.seed < 0:
        args.seed = int(time.time())
    return args


class RunResult(TypedDict):
    """Result from a simulation run."""

    time_spent: float
    """Total wall-clock time."""
    sim_progress: float
    """Finished timeline progress, 1.0 means all simulation finished."""
    requests: Any
    """Per-request statistics."""
    nodes: Any
    """Per-node statistics."""
