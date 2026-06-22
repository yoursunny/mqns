"""
Simulate a linear quantum repeater network to validate model-driven wait-time budgets.

This script demonstrates how to configure ``CutoffSchemeWaitTime`` constraints dynamically
across a 3-node linear topology (``S``, ``R``, ``D``) using predetermined wait-time budgets
(``ROW_INPUTS``). The core objective is to enforce target minimum end-to-end entanglement
fidelities (``F_req``) and contrast the empirical simulation output against closed-form
theoretical model predictions.

The simulation executes in a two-stage sequential pipeline:

1. Calibration Phase (``run_calibration``)
   Executes a baseline simulation without entanglement swapping to calculate the steady-state
   elementary LinkLayer arrival rates (``lam_SR``, ``lam_RD``) across individual quantum fiber links.

2. Evaluation Phase (``run_evaluation``)
   Executes a proactive, centralized simulation using active time-out windows (``swap_cutoff``)
   at the intermediate repeater. Using the calibrated link arrival rates, it maps empirical
   results directly alongside mathematical model estimations for both throughput and fidelity.

Collected Statistics and Outputs:
* **End-to-End Entanglement Rate:** Measured in entanglements per second (Hz).
* **Fidelity Metrics:** Tracked as a macro-average across multiple stochastic runs.
  Runs with zero end-to-end deliveries are automatically isolated via ``np.nanmean`` to prevent
  skewing the average.

The script aggregates results into a multi-tiered dictionary format structured as a ``Report`` schema,
which saves directly to a JSON payload (``--json``) and compiles a dual-axis line plot
(``--plt``) mapping the throughput-fidelity tradeoff curves.
"""

import itertools
import json
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from typing import TypedDict, override

import numpy as np
from tap import Tap

from mqns.entity.qchannel import LinkArchDimDual
from mqns.models.epr import MixedStateEntanglement
from mqns.network.builder import CTRL_DELAY, ChannelParam, NetworkBuilder, NodeDef, tap_configure
from mqns.network.network import QuantumNetwork
from mqns.network.protocol.consumer import RequestCounters
from mqns.network.protocol.link_layer import LinkLayer
from mqns.simulator import Simulator
from mqns.utils import log, rng, unwrap

from examples_common.plotting import plt, plt_save

log.set_default_level("CRITICAL")


class Args(Tap):
    workers: int = 1  # number of workers for parallel execution
    runs: int = 10  # number of trials per parameter set
    sim_duration: tuple[float, float] = (5.0, 25.0)  # calibration and evaluation duration in seconds
    json: str = ""  # save report as JSON file
    plt: str = ""  # save plot as image file

    @override
    def configure(self) -> None:
        tap_configure(self)

    def to_json(self) -> dict:
        """Summarize input parameters to include in JSON output."""
        return {
            "runs": self.runs,
            "sim_duration": self.sim_duration,
        }


SIMULATOR_ACCURACY = 1000000
SEED_BASE = 100


@dataclass
class RowInput:
    f_req: float
    """Required fidelity."""
    w: tuple[float, float]
    """Wait-time budgets."""


ROW_INPUTS: list[RowInput] = [
    RowInput(0.75, (0.031731, 0.023798)),
    RowInput(0.7823, (0.023991, 0.017993)),
    RowInput(0.8146, (0.017056, 0.012792)),
    RowInput(0.8469, (0.010775, 0.008081)),
    RowInput(0.8791, (0.005053, 0.003789)),
    RowInput(0.8904, (0.003155, 0.002367)),
    RowInput(0.9001, (0.001569, 0.001177)),
    RowInput(0.9098, (1.9e-05, 1.4e-05)),
]


class Stats(TypedDict):
    rate: float
    """Rate -- number of entanglements consumed per second."""
    fid_mean: float
    """Fidelity -- mean."""
    fid_std: float
    """Fidelity -- standard deviation."""


class Row(TypedDict):
    """Evaluation result for one f_req, together with model estimation."""

    f_req: float
    """Required fidelity."""
    rate_mdl: float
    """Rate -- model estimation."""
    rate_mean: float
    """Rate -- mean."""
    rate_std: float
    """Rate -- standard deviation."""
    fid_mdl: float
    """Fidelity -- model estimation."""
    fid_mean: float
    """Fidelity -- mean of per-run means."""
    fid_std: float
    """Fidelity -- standard deviation of per-run means."""
    runs: list[Stats]
    """Per-run statistics."""


class Report(TypedDict):
    """JSON report."""

    args: dict
    """Input arguments summary."""
    lam_mean: list[float]
    """LinkLayer arrival rate for each link -- mean from calibration."""
    lam_std: list[float]
    """LinkLayer arrival rate for each link -- standard deviation from calibration."""
    lam_runs: list[list[float]]
    """LinkLayer arrival rate for each link -- per calibration run."""
    eval: list[Row]
    """Evaluation results."""


def convert_fidelity(raw_fidelity: float, x_ratio: float, y_ratio: float, z_ratio: float):
    p_error = 1 - raw_fidelity
    return raw_fidelity, p_error * z_ratio, p_error * x_ratio, p_error * y_ratio


def run_simulation(seed: int, args: Args, duration: float, w: tuple[float, float] | None) -> QuantumNetwork:
    _ = args
    rng.reseed(seed)

    net = (
        NetworkBuilder(
            epr_type=MixedStateEntanglement,
        )
        .topo_linear(
            nodes=[
                NodeDef("S", t_cohere=1 / 5),
                "R",
                "D",
            ],
            t_cohere=1 / 10,
            channels=[
                ChannelParam(ch_length=32, init_fidelity=convert_fidelity(0.9474, 0.1427, 0.1427, 0.7147)),
                ChannelParam(ch_length=18, init_fidelity=convert_fidelity(0.9677, 0.1547, 0.1547, 0.6907)),
            ],
            fiber_alpha=0.2,
            link_arch=LinkArchDimDual,
            eta_d=0.58,
            eta_s=0.99,
            frequency=80e6,
            tau_0=10e-6,
        )
        .proactive_centralized(
            p_swap=0.5,
            swap_delay=340e-6,
            swap_error="PERFECT",  # no error applied by the swap gates
            swap_error_at="f",  # memory decoherence continues during swap
        )
        .request(
            "S-D",
            swap="disabled" if w is None else "asap",
            swap_cutoff=w,
        )
        .make_network()
    )

    RequestCounters.enable_collect_all(net, 0, "S-D")

    s = Simulator(0, duration + CTRL_DELAY, accuracy=SIMULATOR_ACCURACY, install_to=(log, net))
    s.run()

    return net


def run_calibration(seed: int, args: Args) -> list[float]:
    duration = args.sim_duration[0]
    net = run_simulation(seed, args, duration, None)

    lam: list[float] = []
    for link_primary in "S", "R":
        ll_cnt = net.get_node(link_primary).get_app(LinkLayer).cnt
        lam.append(ll_cnt.n_etg / duration)
    return lam


def run_evaluation(seed: int, args: Args, ri: RowInput):
    duration = args.sim_duration[1]
    net = run_simulation(seed, args, duration, ri.w)

    req_cnt = RequestCounters.of(net, 0, "S-D")
    fids = np.array(unwrap(req_cnt.consumed_fidelity_values), dtype=float)
    if len(fids) == 0:
        fids = np.array([0])
    return [
        Stats(
            rate=req_cnt.get_rate(duration),
            fid_mean=fids.mean(),
            fid_std=fids.std(),
        )
    ]


def run_model(lam: list[float], w: tuple[float, float]) -> tuple[float, float]:
    def simple_rate(lam1, T1, W1, lam2, T2, W2, Tswp):
        a1 = 1 - np.exp(-lam2 * W1)
        a2 = 1 - np.exp(-lam1 * W2)
        Tpair = max(T1, T2) + Tswp
        E12 = lam1 * a1
        E21 = lam2 * a2

        R = (E12 + E21) / (
            1 + E12 * (1 / lam2 + Tpair) + 1 * lam1 * (1 - a1) * T1 + E21 * (1 / lam1 + Tpair) + 1 * lam2 * (1 - a2) * T2
        )
        return R

    def h(a):
        return a + (1 - a) * np.log(1 - a) if a < 1 else 1

    def waittimes2swap(lambdas, Ws):
        lam1, lam2 = lambdas
        W1, W2 = Ws

        a1 = 1 - np.exp(-lam2 * W1)
        a2 = 1 - np.exp(-lam1 * W2)

        D = a1 * lam1 + a2 * lam2

        return [h(a1) * lam1 / lam2 / D, h(a2) * lam2 / lam1 / D]

    q = 0.5
    v_swp = 0.9506
    w_swp = 0.9048374180359596
    A0 = 0.018300000000000004
    gam01 = 15
    gam12 = 20
    T1 = 0.00017
    T2 = 0.0001
    Tswp = 0.00034

    def waittime2fidelity(x):
        return (1 + v_swp * (1 + 2 * w_swp * np.exp(-(A0 if x is None else A0 + gam01 * x[0] + gam12 * x[1])))) / 4

    R = q * simple_rate(lam[0], T1, w[0], lam[1], T2, w[1], Tswp)
    F = waittime2fidelity(waittimes2swap(lam, w))

    return R, F


def run_row(args: Args, ri: RowInput, lam_mean: list[float]) -> Row:
    rate_mdl, fid_mdl = run_model(lam_mean, ri.w)

    runs: list[Stats] = []
    for i in range(args.runs):
        runs += run_evaluation(SEED_BASE + i, args, ri)

    rates = np.fromiter((s["rate"] for s in runs), dtype=float)
    fids = np.fromiter((s["fid_mean"] if s["rate"] > 0 else np.nan for s in runs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fid_mean, fid_std = np.nanmean(fids).item(), np.nanstd(fids).item()

    return Row(
        f_req=ri.f_req,
        rate_mdl=rate_mdl,
        rate_mean=rates.mean(),
        rate_std=rates.std(),
        fid_mdl=fid_mdl,
        fid_mean=fid_mean,
        fid_std=fid_std,
        runs=runs,
    )


def main(args: Args) -> Report:
    # Run calibration step to determine LinkLayer arrival rates of each channel.
    with Pool(processes=args.workers) as pool:
        lam_runs = pool.starmap(run_calibration, itertools.product(range(SEED_BASE, SEED_BASE + args.runs), [args]))
    lam_mean: list[float] = []
    lam_std: list[float] = []
    # Collect mean LinkLayer arrival rates of each channel.
    for i in range(len(lam_runs[0])):  # len(lam_runs[0]) is number of links in the linear network
        lam_array = np.fromiter((lam_run[i] for lam_run in lam_runs), dtype=float)
        lam_mean.append(lam_array.mean())
        lam_std.append(lam_array.std())

    # Run evaluation step to determine end-to-end rate and fidelity.
    # Theoretical rate and fidelity are also calculated based on desired fidelity and calibrated LinkLayer arrival rates.
    with Pool(processes=args.workers) as pool:
        rows = pool.starmap(run_row, itertools.product([args], ROW_INPUTS, [lam_mean]))

    # Generate report.
    return Report(
        args=args.to_json(),
        lam_mean=lam_mean,
        lam_std=lam_std,
        lam_runs=lam_runs,
        eval=rows,
    )


def plot(args: Args, rows: list[Row]) -> None:
    F_req = [row["f_req"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(5, 4), constrained_layout=True)
    color = "tab:blue"
    ax1.set_xlabel("Required Fidelity")
    ax1.set_ylabel("Rate (epps)", color=color)
    ax1.plot(F_req, [row["rate_mdl"] for row in rows], color=color, label="Rate (Model)")
    ax1.errorbar(
        F_req,
        [row["rate_mean"] for row in rows],
        yerr=[row["rate_std"] for row in rows],
        fmt="d",
        color=color,
        label="Simulated rate",
        markersize=4,
        capsize=3,
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Fidelity", color=color)
    ax2.plot(F_req, [row["fid_mdl"] for row in rows], color=color, label="Fidelity (Model)")
    ax2.errorbar(
        F_req,
        [row["fid_mean"] for row in rows],
        yerr=[row["fid_std"] for row in rows],
        fmt="d",
        color=color,
        label="Simulated fidelity",
        markersize=4,
        capsize=3,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Fidelity vs Rate tradeoff", fontsize="medium")
    fig.legend(loc="center left", bbox_to_anchor=(0.12, 0.5))
    plt.grid()
    plt_save(args.plt)


if __name__ == "__main__":
    freeze_support()
    args = Args().parse_args()
    report = main(args)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file)
    plot(args, report["eval"])
