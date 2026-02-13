import json
from typing import cast

import numpy as np
from tap import Tap

from mqns.network.builder import CTRL_DELAY, NetworkBuilder
from mqns.network.proactive import MuxScheme, MuxSchemeDynamicEpr, MuxSchemeStatistical, ProactiveForwarder
from mqns.simulator import Simulator
from mqns.utils import log, rng

from examples_common.plotting import Axes2D, mpl, plt, plt_save


class Args(Tap):
    runs: int = 3  # number of trials per parameter set
    json: str = ""  # save results as JSON file
    plt: str = ""  # save plot as image file


args = Args().parse_args()

log.set_default_level("CRITICAL")

SEED_BASE = 100

# parameters
sim_duration = 3

# Quantum channel lengths
ch_S1_R1 = 10
ch_R1_R2 = 10
ch_R2_R3 = 10
ch_R3_D1 = 10
ch_S2_R2 = 10
ch_R3_D2 = 10


def run_simulation(t_cohere: float, mux: MuxScheme, seed: int):
    rng.reseed(seed)

    net = (
        NetworkBuilder()
        .topo(
            channels=[
                ("S1-R1", ch_S1_R1),
                ("R1-R2", ch_R1_R2),
                ("R2-R3", ch_R2_R3),
                ("R3-D1", ch_R3_D1),
                ("s2-R2", ch_S2_R2),
                ("R3-D2", ch_R3_D2),
            ],
            t_cohere=t_cohere,
        )
        .proactive_centralized(mux=mux)
        .path(src="S1", dst="D1", swap="asap")
        .path(src="S2", dst="D2", swap="asap")
        .make_network()
    )

    s = Simulator(0, sim_duration + CTRL_DELAY, accuracy=1000000, install_to=(log, net))
    s.run()

    #### get stats: e2e_rate and mean_fidelity
    fw_s1 = net.get_node("S1").get_app(ProactiveForwarder)
    fw_s2 = net.get_node("S2").get_app(ProactiveForwarder)
    # [(path 1), (path 2), ...]
    return [
        (fw_s1.cnt.n_consumed / sim_duration, fw_s1.cnt.consumed_avg_fidelity),
        (fw_s2.cnt.n_consumed / sim_duration, fw_s2.cnt.consumed_avg_fidelity),
    ]


# Simulation constants
SEED_BASE = 100
t_cohere_values = [5e-3, 10e-3, 20e-3]

# Strategy configs
strategies: dict[str, MuxScheme] = {
    "Statistical Mux.": MuxSchemeStatistical(),
    "Random Alloc.": MuxSchemeDynamicEpr(),
    "Swap-weighted Alloc.": MuxSchemeDynamicEpr(select_path=MuxSchemeDynamicEpr.SelectPath_swap_weighted),
}


results = {strategy: {0: [], 1: []} for strategy in strategies}

# Run simulation
for strategy, mux in strategies.items():
    for t_cohere in t_cohere_values:
        path_rates: list[list[float]] = [[], []]
        path_fids: list[list[float]] = [[], []]
        for i in range(args.runs):
            print(f"{strategy}, T_cohere={t_cohere:.3f}, run #{i}")
            seed = SEED_BASE + i
            (rate1, fid1), (rate2, fid2) = run_simulation(t_cohere, mux, seed)
            path_rates[0].append(rate1)
            path_rates[1].append(rate2)
            path_fids[0].append(fid1)
            path_fids[1].append(fid2)
        for path in [0, 1]:
            mean_rate = np.mean(path_rates[path])
            std_rate = np.std(path_rates[path])
            mean_fid = np.mean(path_fids[path])
            std_fid = np.std(path_fids[path])
            results[strategy][path].append((mean_rate, std_rate, mean_fid, std_fid))


if args.json:
    with open(args.json, "w") as file:
        json.dump(results, file)

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 22,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "errorbar.capsize": 4,
    }
)

PATH_TITLES = ("S1-D1", "S2-D2")
fig, axs = plt.subplots(nrows=2, ncols=len(PATH_TITLES), figsize=(9, 8), sharex=True, sharey="row")
axs = cast(Axes2D, axs)

# Plot Entanglement Rate
for strategy in strategies:
    for path, path_title in enumerate(PATH_TITLES):
        rates = [results[strategy][path][i][0] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][1] for i in range(len(t_cohere_values))]
        ax = axs[0, path]
        ax.errorbar(
            [t * 1e3 for t in t_cohere_values],
            rates,
            yerr=stds,
            marker="o",
            label=strategy,
        )
        ax.set_title(path_title)
        ax.set_ylabel("E2E Rate (eps)")
        ax.grid(True)

# Plot Fidelity
for strategy in strategies:
    for path, path_title in enumerate(PATH_TITLES):
        fids = [results[strategy][path][i][2] for i in range(len(t_cohere_values))]
        stds = [results[strategy][path][i][3] for i in range(len(t_cohere_values))]
        ax = axs[1, path]
        ax.errorbar(
            [t * 1e3 for t in t_cohere_values],
            fids,
            yerr=stds,
            marker="s",
            label=strategy,
        )
        ax.set_title(path_title)
        ax.set_xlabel("T_cohere (ms)")
        ax.set_ylabel("Fidelity")
        ax.grid(True)

axs[1, 1].legend(title="Strategy", loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt_save(args.plt)
