import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS = 1.0e-12
State4 = tuple[int, int, int, int]  # (a,p,b1,b2)


def normalize_probs(probs):
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    else:
        return [0 for p in probs]


def channel2Pauli(depol, depha):
    v0 = 1 - depol
    w0 = 1 - depha
    pI = (1 + v0 * (1 + 2 * w0)) / 4
    pX = (1 - v0) / 4
    pY = (1 - v0) / 4
    pZ = (1 + v0 * (1 - 2 * w0)) / 4
    return pI, normalize_probs((pX, pY, pZ))


def simple_rate(lam1, T1, W1, lam2, T2, W2, Tswp=0):
    """
    Calculates the swapping rate for two queues with
    Poisson arrivals, unit capacity, deadlines W, and reset delays T.
    Uses a simplified approximation based on the expected match time and success probabilities.

    Parameters:
    lam1, lam2 : float : Arrival rates for Queue 1 and Queue 2
    W1, W2     : float : Item lifetimes (max wait time) for Queue 1 and Queue 2
    T1, T2     : float : Reset delays for Queue 1 and Queue 2
    Tswp       : float : Additional swapping time (default=0)

    Returns:
    float : Expected rate.
    """
    a1 = 1 - np.exp(-lam2 * W1)
    a2 = 1 - np.exp(-lam1 * W2)
    Tpair = max(T1, T2) + Tswp
    E12 = lam1 * a1
    E21 = lam2 * a2

    R = (E12 + E21) / (
        1 + E12 * (1 / lam2 + Tpair) + 1 * lam1 * (1 - a1) * T1 + E21 * (1 / lam1 + Tpair) + 1 * lam2 * (1 - a2) * T2
    )
    return R


def waittimes2swap(lambdas, Ws):
    # This function calculate expected wait times to swap of the 2 queues
    # lambdas:  arrival rates in the unit-capacity case
    # Ws: cutoff times of the queues.

    lam1, lam2 = lambdas
    W1, W2 = Ws

    a1 = 1 - np.exp(-lam2 * W1)
    a2 = 1 - np.exp(-lam1 * W2)

    # h = lambda a: a + (1-a)*np.log(1-a)  # entropy-like function for the expected time calculation
    def h(a):
        # modified to avoid log(0) when a=1, which can happen when W is large and lam is high.
        # In that case, we can treat h(a)=1 since it's the limit as a->1.
        return a + (1 - a) * np.log(1 - a) if a < 1 else 1

    D = a1 * lam1 + a2 * lam2
    wait_times = [h(a1) * lam1 / lam2 / D, h(a2) * lam2 / lam1 / D]

    return wait_times


class QueueSpec(NamedTuple):
    M: int
    """How many memory qubit pairs on this link."""
    lam: float | list[float]
    """Entanglement arrival rate in Hz for each qubit pair."""
    tau: Any
    """Unused."""
    W: float
    """Wait-time budget in seconds."""
    T: float
    """Interval between consecutive entanglement generation attempts."""
    f: Any
    """Unused."""


def swap_2Q(
    Q1: QueueSpec,
    Q2: QueueSpec,
    Tswp: float = 0,
    method: Literal["BD", "CTMC", "SIM"] = "BD",
    capacity: Literal["single", "multi"] = "multi",
    matching_policy: Literal["FIFO", "LIFO", "RANDOM", "RAND"] = "RANDOM",
) -> tuple[float, list[float]]:
    """
    Calculates the swapping rate for two queues with
    Poisson arrivals, unit capacity, deadlines W, and reset delays T.

    Parameters:
    Q1, Q2 : tuple (Q, lambda, tau, W, T, f) for Queue 1 and Queue 2
    Tswp    : Additional swapping time (default=0)
    method: use model CTMC, BD, or SIM (MonteCarlo simulation)
    policy  : must be FIFO, LIFO, or RANDOM
    Returns:
    Expected rate, expected wait times until swap.
    """
    m1, lam1_list, _, W1, T1, f1 = Q1  # Renamed lam1 to lam1_list to reflect its type
    m2, lam2_list, _, W2, T2, f2 = Q2  # Renamed lam2 to lam2_list to reflect its type

    T2 = 10e-6  # Node 1 is the initiator
    mem = [m1, m2]
    if mem == [1, 1] and capacity in ["single", "unit", "1"]:  # use simple formula for single memory case
        # Extract scalar rates for single memory case
        scalar_lam1 = lam1_list[0] if isinstance(lam1_list, list) else lam1_list
        scalar_lam2 = lam2_list[0] if isinstance(lam2_list, list) else lam2_list

        R = simple_rate(lam1=scalar_lam1, T1=T1, W1=W1, lam2=scalar_lam2, T2=T2, W2=W2, Tswp=Tswp)
        wait_times = waittimes2swap((scalar_lam1, scalar_lam2), (W1, W2))

        return R, wait_times

    else:  # for multiple memories
        # set up rate vectors:
        lambda1 = lam1_list if isinstance(lam1_list, list) else [lam1_list] * m1  # Ensure it's a list for CTMC
        lambda2 = lam2_list if isinstance(lam2_list, list) else [lam2_list] * m2  # Ensure it's a list for CTMC

        res = solve_matching(mem[0], mem[1], lambda1, lambda2, W1, W2, Tswp, T1, T2, matching_policy, model=method.lower())
        return res["expected_matching_rate"], [res["expected_wait_time_EZ1"], res["expected_wait_time_EZ2"]]


# ============================================================================


def _normalize_policy(policy: str) -> str:
    p = policy.strip().lower()
    if p in {"fifo", "first-in-first-out"}:
        return "fifo"
    if p in {"lifo", "last-in-first-out"}:
        return "lifo"
    if p in {"random", "rand"}:
        return "random"
    raise ValueError("policy must be FIFO, LIFO, or RANDOM")


def _parse_rates(lam: Sequence[float], K: int, name: str) -> list[float]:
    vals = [float(x) for x in lam]
    if len(vals) == 1:
        vals = vals * K + [0.0]
    elif len(vals) == K:
        vals = vals + [0.0]
    elif len(vals) == K + 1:
        vals = vals[:]
        vals[K] = 0.0
    else:
        raise ValueError(f"{name} must have length 1, K, or K+1")
    if any(x < 0 for x in vals):
        raise ValueError(f"{name} entries must be nonnegative")
    return vals


def _safe_rate_from_mean(x: float, name: str) -> float:
    if x <= 0 or not math.isfinite(x):
        raise ValueError(f"{name} must be positive and finite")
    return 1.0 / float(x)


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > EPS else math.nan


# ---------------------------------------------------------------------------
# Birth-death approximation
# ---------------------------------------------------------------------------


@dataclass
class BDMetrics:
    pi: dict[int, float]
    rate: float
    R1_wait: float
    R2_wait: float
    s1: float
    s2: float
    theta1: float
    theta2: float


def _bd_up_rate(b: int, K1: int, K2: int, lam1: list[float], lam2: list[float], theta1: float, theta2: float) -> float:
    if b < 0:
        return lam2[0] + (-b) * theta1
    if b < K2:
        return lam2[b]
    return 0.0


def _bd_down_rate(b: int, K1: int, K2: int, lam1: list[float], lam2: list[float], theta1: float, theta2: float) -> float:
    if b == -K1:
        return 0.0
    if b <= 0:
        return lam1[-b]
    return lam1[0] + b * theta2


def _bd_stationary(K1: int, K2: int, lam1: list[float], lam2: list[float], theta1: float, theta2: float) -> dict[int, float]:
    weights: dict[int, float] = {0: 1.0}
    w = 1.0
    for b in range(1, K2 + 1):
        num = _bd_up_rate(b - 1, K1, K2, lam1, lam2, theta1, theta2)
        den = _bd_down_rate(b, K1, K2, lam1, lam2, theta1, theta2)
        w *= num / den if den > 0 else 0.0
        weights[b] = w
    w = 1.0
    for n in range(1, K1 + 1):
        prev = -(n - 1)
        cur = -n
        num = _bd_down_rate(prev, K1, K2, lam1, lam2, theta1, theta2)
        den = _bd_up_rate(cur, K1, K2, lam1, lam2, theta1, theta2)
        w *= num / den if den > 0 else 0.0
        weights[cur] = w
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError("birth-death stationary distribution is degenerate")
    return {b: weights[b] / total for b in weights}


def _bd_metrics(K1: int, K2: int, lam1: list[float], lam2: list[float], W1: float, W2: float) -> BDMetrics:
    theta1 = 1.0 / W1
    theta2 = 1.0 / W2
    pi = _bd_stationary(K1, K2, lam1, lam2, theta1, theta2)
    R1 = sum(pi.get(-u, 0.0) * lam2[0] for u in range(1, K1 + 1))
    R2 = sum(pi.get(u, 0.0) * lam1[0] for u in range(1, K2 + 1))
    s1 = sum(pi.get(-u, 0.0) * lam1[u] for u in range(0, K1))
    s2 = sum(pi.get(u, 0.0) * lam2[u] for u in range(0, K2))
    return BDMetrics(pi=pi, rate=R1 + R2, R1_wait=R1, R2_wait=R2, s1=s1, s2=s2, theta1=theta1, theta2=theta2)


# ---------------------------------------------------------------------------
# 4D count CTMC
# ---------------------------------------------------------------------------


@dataclass
class CTMCMetrics:
    states: list[State4]
    index: dict[State4, int]
    pi: np.ndarray
    rate: float
    R1_wait: float
    R2_wait: float
    s1: float
    s2: float
    theta1: float
    theta2: float
    K1: int
    K2: int
    lam1: list[float]
    lam2: list[float]
    gamma: float
    delta1: float
    delta2: float


def _alive_count(s: State4, side: int) -> int:
    a, _p, _b1, _b2 = s
    return max(-a, 0) if side == 1 else max(a, 0)


def _N(s: State4, side: int) -> int:
    a, p, b1, b2 = s
    if side == 1:
        return max(-a, 0) + p + b1
    return max(a, 0) + p + b2


def _enumerate_ctmc_states(K1: int, K2: int) -> list[State4]:
    states: list[State4] = []
    for p in range(min(K1, K2) + 1):
        for b1 in range(K1 - p + 1):
            room1 = K1 - p - b1
            for b2 in range(K2 - p + 1):
                room2 = K2 - p - b2
                for a in range(-room1, room2 + 1):
                    states.append((a, p, b1, b2))
    states.sort()
    return states


def _ctmc_metrics(
    K1: int, K2: int, lam1: list[float], lam2: list[float], W1: float, W2: float, Tp: float, T1: float, T2: float
) -> CTMCMetrics:
    theta1, theta2 = 1.0 / W1, 1.0 / W2
    gamma = _safe_rate_from_mean(Tp, "Tp")
    delta1 = _safe_rate_from_mean(T1, "T1")
    delta2 = _safe_rate_from_mean(T2, "T2")
    states = _enumerate_ctmc_states(K1, K2)
    idx = {s: i for i, s in enumerate(states)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    out = np.zeros(len(states))

    def add(src_i: int, dst: State4, rate: float) -> None:
        if rate <= EPS:
            return
        rows.append(src_i)
        cols.append(idx[dst])
        data.append(rate)
        out[src_i] += rate

    for i, s in enumerate(states):
        a, p, b1, b2 = s
        N1, N2 = _N(s, 1), _N(s, 2)
        # side 1 arrival
        if N1 < K1 and lam1[N1] > 0:
            if a > 0:  # matches waiting side 2
                add(i, (a - 1, p + 1, b1, b2), lam1[N1])
            else:
                add(i, (a - 1, p, b1, b2), lam1[N1])
        # side 2 arrival
        if N2 < K2 and lam2[N2] > 0:
            if a < 0:  # matches waiting side 1
                add(i, (a + 1, p + 1, b1, b2), lam2[N2])
            else:
                add(i, (a + 1, p, b1, b2), lam2[N2])
        # expiration of alive items
        if a < 0:
            add(i, (a + 1, p, b1 + 1, b2), (-a) * theta1)
        elif a > 0:
            add(i, (a - 1, p, b1, b2 + 1), a * theta2)
        # processing completion
        if p > 0:
            add(i, (a, p - 1, b1 + 1, b2 + 1), p * gamma)
        # delay exits
        if b1 > 0:
            add(i, (a, p, b1 - 1, b2), b1 * delta1)
        if b2 > 0:
            add(i, (a, p, b1, b2 - 1), b2 * delta2)

    for i, val in enumerate(out):
        rows.append(i)
        cols.append(i)
        data.append(-val)
    Q = sp.csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    A = Q.T.tolil()
    rhs = np.zeros(len(states))
    A[-1, :] = np.ones(len(states))
    rhs[-1] = 1.0
    pi = spla.spsolve(A.tocsr(), rhs)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()

    R1 = 0.0  # waiting side 1 matched by side 2 arrival
    R2 = 0.0  # waiting side 2 matched by side 1 arrival
    s1 = 0.0  # non-immediate side 1 arrivals
    s2 = 0.0
    for i, s in enumerate(states):
        a, p, b1, b2 = s
        N1, N2 = _N(s, 1), _N(s, 2)
        l1 = lam1[N1] if N1 < K1 else 0.0
        l2 = lam2[N2] if N2 < K2 else 0.0
        if a < 0:
            R1 += pi[i] * l2
        if a > 0:
            R2 += pi[i] * l1
        # non-immediate arrivals join if other side has no alive waiting
        if a <= 0:
            s1 += pi[i] * l1
        if a >= 0:
            s2 += pi[i] * l2

    return CTMCMetrics(
        states=states,
        index=idx,
        pi=pi,
        rate=R1 + R2,
        R1_wait=R1,
        R2_wait=R2,
        s1=s1,
        s2=s2,
        theta1=theta1,
        theta2=theta2,
        K1=K1,
        K2=K2,
        lam1=lam1,
        lam2=lam2,
        gamma=gamma,
        delta1=delta1,
        delta2=delta2,
    )


# ---------------------------------------------------------------------------
# Latent tagged chains for BD and CTMC
# ---------------------------------------------------------------------------


@dataclass
class TaggedChain:
    H: sp.csr_matrix
    alpha: np.ndarray
    states: list[Any]


def _latent_truncated_mean(H: sp.csr_matrix, alpha: np.ndarray, W: float) -> float:
    """E[T | T <= W] for a latent chain with one absorbing match state."""
    n = H.shape[0]
    if n == 0:
        return math.nan
    A = (-H).tocsr()
    one = np.ones(n)
    surv_vec = spla.expm_multiply(H * W, one)
    survival = float(alpha @ surv_vec)
    L = max(0.0, min(1.0, 1.0 - survival))
    if L <= EPS:
        return math.nan
    integ_vec = spla.spsolve(A, one - surv_vec)
    integral_survival = float(alpha @ integ_vec)
    G = integral_survival - W * survival
    return max(0.0, G / L)


def _build_bd_latent_chain(
    K1: int, K2: int, lam1: list[float], lam2: list[float], met: BDMetrics, side: int, policy: str
) -> TaggedChain:
    policy = _normalize_policy(policy)
    if side == 1:
        K, lam_same, eta, theta = K1, lam1, lam2[0], met.theta1
    else:
        K, lam_same, eta, theta = K2, lam2, lam1[0], met.theta2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    if policy == "random":
        states: list[Any] = list(range(1, K + 1))
        index = {u: i for i, u in enumerate(states)}
        for i, u in enumerate(states):
            out = 0.0
            if u < K and lam_same[u] > 0:
                rows.append(i)
                cols.append(index[u + 1])
                data.append(lam_same[u])
                out += lam_same[u]
            # opposite arrival: tag matched with prob 1/u; non-tag removed otherwise
            out += eta / u
            if u > 1:
                rate = eta * (u - 1) / u + (u - 1) * theta
                rows.append(i)
                cols.append(index[u - 1])
                data.append(rate)
                out += rate
            rows.append(i)
            cols.append(i)
            data.append(-out)
        H = sp.csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
        alpha = np.zeros(len(states))
        s = met.s1 if side == 1 else met.s2
        if s > 0:
            for u0 in range(0, K):
                b = -u0 if side == 1 else u0
                flow = met.pi.get(b, 0.0) * lam_same[u0]
                alpha[index[u0 + 1]] += flow / s
        return TaggedChain(H=H, alpha=alpha, states=states)

    states = [(u, k) for u in range(1, K + 1) for k in range(u)]
    index = {s: i for i, s in enumerate(states)}
    for i, (u, k) in enumerate(states):
        out = 0.0
        if policy == "fifo":
            if u < K and lam_same[u] > 0:
                rows.append(i)
                cols.append(index[(u + 1, k)])
                data.append(lam_same[u])
                out += lam_same[u]
            if k == 0:
                out += eta  # absorbing match
            else:
                rate = eta + k * theta
                rows.append(i)
                cols.append(index[(u - 1, k - 1)])
                data.append(rate)
                out += rate
            younger = u - 1 - k
            if younger > 0:
                rate = younger * theta
                rows.append(i)
                cols.append(index[(u - 1, k)])
                data.append(rate)
                out += rate
        elif policy == "lifo":
            if u < K and lam_same[u] > 0:
                rows.append(i)
                cols.append(index[(u + 1, k + 1)])
                data.append(lam_same[u])
                out += lam_same[u]
            if k == 0:
                out += eta
            else:
                rate = eta + k * theta
                rows.append(i)
                cols.append(index[(u - 1, k - 1)])
                data.append(rate)
                out += rate
            older = u - 1 - k
            if older > 0:
                rate = older * theta
                rows.append(i)
                cols.append(index[(u - 1, k)])
                data.append(rate)
                out += rate
        rows.append(i)
        cols.append(i)
        data.append(-out)
    H = sp.csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    alpha = np.zeros(len(states))
    s = met.s1 if side == 1 else met.s2
    if s > 0:
        for u0 in range(0, K):
            b = -u0 if side == 1 else u0
            flow = met.pi.get(b, 0.0) * lam_same[u0]
            k0 = u0 if policy == "fifo" else 0
            alpha[index[(u0 + 1, k0)]] += flow / s
    return TaggedChain(H=H, alpha=alpha, states=states)


def _ctmc_state_after(metrics: CTMCMetrics, s: State4, da=0, dp=0, db1=0, db2=0) -> State4:
    ns = (s[0] + da, s[1] + dp, s[2] + db1, s[3] + db2)
    if ns not in metrics.index:
        raise RuntimeError(f"Invalid CTMC tagged transition {s} -> {ns}")
    return ns


def _build_ctmc_latent_chain(metrics: CTMCMetrics, side: int, policy: str) -> TaggedChain:
    policy = _normalize_policy(policy)
    states: list[Any] = []
    index: dict[Any, int] = {}
    # transient states contain a count state with a live tag on side
    for si, s in enumerate(metrics.states):
        m = _alive_count(s, side)
        if m <= 0:
            continue
        if policy == "random":
            key = (si,)
            index[key] = len(states)
            states.append(key)
        else:
            for k in range(m):
                key = (si, k)
                index[key] = len(states)
                states.append(key)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(row: int, dst_key: Any, rate: float) -> None:
        if rate <= EPS:
            return
        if dst_key not in index:
            raise RuntimeError(f"Missing tagged destination {dst_key}")
        rows.append(row)
        cols.append(index[dst_key])
        data.append(rate)

    for key, row in index.items():
        si = key[0]
        k = key[1] if len(key) > 1 else None
        s = metrics.states[si]
        a, p, b1, b2 = s
        m = _alive_count(s, side)
        N1, N2 = _N(s, 1), _N(s, 2)
        out = 0.0
        lam_same = (
            metrics.lam1[N1] if side == 1 and N1 < metrics.K1 else metrics.lam2[N2] if side == 2 and N2 < metrics.K2 else 0.0
        )
        # other = 2 if side == 1 else 1
        lam_opp = (
            metrics.lam2[N2] if side == 1 and N2 < metrics.K2 else metrics.lam1[N1] if side == 2 and N1 < metrics.K1 else 0.0
        )
        theta = metrics.theta1 if side == 1 else metrics.theta2

        # same-side arrival: always a birth because the other side has no alive items if tag side alive
        if lam_same > 0:
            dst = _ctmc_state_after(metrics, s, da=(-1 if side == 1 else 1))
            dsi = metrics.index[dst]
            if policy == "fifo":
                add(row, (dsi, k), lam_same)
            elif policy == "lifo":
                add(row, (dsi, k + 1), lam_same)
            else:
                add(row, (dsi,), lam_same)
            out += lam_same

        # opposite-side arrival: match tag or remove non-tag
        if lam_opp > 0:
            dst = _ctmc_state_after(metrics, s, da=(1 if side == 1 else -1), dp=1)
            dsi = metrics.index[dst]
            if policy in ("fifo", "lifo"):
                if k == 0:
                    out += lam_opp  # absorbing match
                else:
                    add(row, (dsi, k - 1), lam_opp)
                    out += lam_opp
            else:
                out += lam_opp / m  # tag matched
                if m > 1:
                    add(row, (dsi,), lam_opp * (m - 1) / m)
                    out += lam_opp * (m - 1) / m

        # non-tag expiration on tag side. Tag itself does not expire.
        if m > 1:
            dst = _ctmc_state_after(
                metrics, s, da=(1 if side == 1 else -1), db1=(1 if side == 1 else 0), db2=(1 if side == 2 else 0)
            )
            dsi = metrics.index[dst]
            if policy == "fifo":
                older = k
                younger = m - 1 - k
                if older > 0:
                    add(row, (dsi, k - 1), older * theta)
                    out += older * theta
                if younger > 0:
                    add(row, (dsi, k), younger * theta)
                    out += younger * theta
            elif policy == "lifo":
                younger = k
                older = m - 1 - k
                if younger > 0:
                    add(row, (dsi, k - 1), younger * theta)
                    out += younger * theta
                if older > 0:
                    add(row, (dsi, k), older * theta)
                    out += older * theta
            else:
                add(row, (dsi,), (m - 1) * theta)
                out += (m - 1) * theta

        # background processing/delay transitions
        if p > 0:
            dst = _ctmc_state_after(metrics, s, dp=-1, db1=1, db2=1)
            dsi = metrics.index[dst]
            add(row, (dsi, k) if policy != "random" else (dsi,), p * metrics.gamma)
            out += p * metrics.gamma
        if b1 > 0:
            dst = _ctmc_state_after(metrics, s, db1=-1)
            dsi = metrics.index[dst]
            add(row, (dsi, k) if policy != "random" else (dsi,), b1 * metrics.delta1)
            out += b1 * metrics.delta1
        if b2 > 0:
            dst = _ctmc_state_after(metrics, s, db2=-1)
            dsi = metrics.index[dst]
            add(row, (dsi, k) if policy != "random" else (dsi,), b2 * metrics.delta2)
            out += b2 * metrics.delta2

        rows.append(row)
        cols.append(row)
        data.append(-out)

    H = sp.csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))

    # Initial distribution for non-immediate arrivals.
    alpha = np.zeros(len(states))
    s_rate = metrics.s1 if side == 1 else metrics.s2
    if s_rate > 0:
        for si, s in enumerate(metrics.states):
            m_old = _alive_count(s, side)
            if _alive_count(s, 3 - side) > 0:
                continue
            N1, N2 = _N(s, 1), _N(s, 2)
            if side == 1:
                if N1 >= metrics.K1:
                    continue
                lam = metrics.lam1[N1]
                dst = _ctmc_state_after(metrics, s, da=-1)
            else:
                if N2 >= metrics.K2:
                    continue
                lam = metrics.lam2[N2]
                dst = _ctmc_state_after(metrics, s, da=1)
            if lam <= 0:
                continue
            dsi = metrics.index[dst]
            flow = metrics.pi[si] * lam
            if policy == "fifo":
                key0 = (dsi, m_old)
            elif policy == "lifo":
                key0 = (dsi, 0)
            else:
                key0 = (dsi,)
            if key0 in index:
                alpha[index[key0]] += flow / s_rate
    return TaggedChain(H=H, alpha=alpha, states=states)


def _solve_bd(K1: int, K2: int, lam1: list[float], lam2: list[float], W1: float, W2: float, policy: str) -> dict[str, float]:
    met = _bd_metrics(K1, K2, lam1, lam2, W1, W2)
    ch1 = _build_bd_latent_chain(K1, K2, lam1, lam2, met, 1, policy)
    ch2 = _build_bd_latent_chain(K1, K2, lam1, lam2, met, 2, policy)
    z1_cond = _latent_truncated_mean(ch1.H, ch1.alpha, W1)
    z2_cond = _latent_truncated_mean(ch2.H, ch2.alpha, W2)
    return {
        "expected_matching_rate": met.rate,
        # E[Z_i] for the queue-i item in a matched pair, including zero
        # when queue i is the triggering arrival.
        "expected_wait_time_EZ1": _safe_div(met.R1_wait, met.rate) * z1_cond,
        "expected_wait_time_EZ2": _safe_div(met.R2_wait, met.rate) * z2_cond,
        "conditional_wait_if_queue1_waited": z1_cond,
        "conditional_wait_if_queue2_waited": z2_cond,
        "num_count_states": K1 + K2 + 1,
        "num_tagged_states_1": len(ch1.states),
        "num_tagged_states_2": len(ch2.states),
    }


def _solve_ctmc(
    K1: int, K2: int, lam1: list[float], lam2: list[float], W1: float, W2: float, Tp: float, T1: float, T2: float, policy: str
) -> dict[str, float]:
    met = _ctmc_metrics(K1, K2, lam1, lam2, W1, W2, Tp, T1, T2)
    ch1 = _build_ctmc_latent_chain(met, 1, policy)
    ch2 = _build_ctmc_latent_chain(met, 2, policy)
    z1_cond = _latent_truncated_mean(ch1.H, ch1.alpha, W1)
    z2_cond = _latent_truncated_mean(ch2.H, ch2.alpha, W2)
    return {
        "expected_matching_rate": met.rate,
        # E[Z_i] for the queue-i item in a matched pair, including zero
        # when queue i is the triggering arrival.
        "expected_wait_time_EZ1": _safe_div(met.R1_wait, met.rate) * z1_cond,
        "expected_wait_time_EZ2": _safe_div(met.R2_wait, met.rate) * z2_cond,
        "conditional_wait_if_queue1_waited": z1_cond,
        "conditional_wait_if_queue2_waited": z2_cond,
        "num_count_states": len(met.states),
        "num_tagged_states_1": len(ch1.states),
        "num_tagged_states_2": len(ch2.states),
    }


# ---------------------------------------------------------------------------
# Deterministic event simulation
# ---------------------------------------------------------------------------


@dataclass
class _SimItem:
    item_id: int
    side: int
    birth_time: float
    status: str = "waiting"


class _DeterministicSimulator:
    def __init__(self, K1, K2, lam1, lam2, W1, W2, Tp, T1, T2, policy, horizon=50000.0, warmup=5000.0, seed=12345):
        self.K = {1: K1, 2: K2}
        self.lam = {1: lam1, 2: lam2}
        self.W = {1: W1, 2: W2}
        self.T = {1: T1, 2: T2}
        self.Tp = Tp
        self.policy = _normalize_policy(policy)
        self.horizon = horizon
        self.warmup = warmup
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.next_id = 0
        self.waiting = {1: [], 2: []}
        self.non_exited = {1: 0, 2: 0}
        self.events: list[tuple[float, int, str, Any]] = []
        self.seq = 0
        self.items: dict[int, _SimItem] = {}
        self.matches = 0
        self.wait_sum = {1: 0.0, 2: 0.0}

    def _push(self, time: float, kind: str, data: Any) -> None:
        self.seq += 1
        heapq.heappush(self.events, (time, self.seq, kind, data))

    def _clean_waiting(self, side: int) -> None:
        self.waiting[side] = [iid for iid in self.waiting[side] if self.items.get(iid) and self.items[iid].status == "waiting"]

    def _select_waiting(self, side: int) -> int | None:
        self._clean_waiting(side)
        if not self.waiting[side]:
            return None
        if self.policy == "fifo":
            return self.waiting[side].pop(0)
        if self.policy == "lifo":
            return self.waiting[side].pop()
        k = int(self.rng.integers(len(self.waiting[side])))
        return self.waiting[side].pop(k)

    def _arrival_rate(self, side: int) -> float:
        n = self.non_exited[side]
        if n >= self.K[side]:
            return 0.0
        return self.lam[side][n]

    def _record_match(self, wait1: float, wait2: float) -> None:
        if self.t >= self.warmup:
            self.matches += 1
            self.wait_sum[1] += wait1
            self.wait_sum[2] += wait2

    def run(self) -> dict[str, float]:
        while self.t < self.horizon:
            rate1 = self._arrival_rate(1)
            rate2 = self._arrival_rate(2)
            total_rate = rate1 + rate2
            next_arrival_time = math.inf
            next_arrival_side = None
            if total_rate > 0:
                next_arrival_time = self.t + float(self.rng.exponential(1.0 / total_rate))
                next_arrival_side = 1 if self.rng.random() < rate1 / total_rate else 2
            next_det_time = self.events[0][0] if self.events else math.inf
            if next_arrival_time <= next_det_time and next_arrival_time <= self.horizon:
                self.t = next_arrival_time
                self._handle_arrival(next_arrival_side)
            elif next_det_time <= self.horizon:
                self.t, _seq, kind, data = heapq.heappop(self.events)
                if kind == "expire":
                    iid = data
                    it = self.items.get(iid)
                    if it is not None and it.status == "waiting":
                        it.status = "delay"
                        self._push(self.t + self.T[it.side], "exit", iid)
                elif kind == "exit":
                    iid = data
                    it = self.items.get(iid)
                    if it is not None and it.status != "exited":
                        it.status = "exited"
                        self.non_exited[it.side] -= 1
                else:
                    raise RuntimeError(kind)
            else:
                self.t = self.horizon
                break
        measure = max(self.horizon - self.warmup, EPS)
        return {
            "expected_matching_rate": self.matches / measure,
            "expected_wait_time_EZ1": self.wait_sum[1] / self.matches if self.matches > 0 else math.nan,
            "expected_wait_time_EZ2": self.wait_sum[2] / self.matches if self.matches > 0 else math.nan,
        }

    def _handle_arrival(self, side: int) -> None:
        if self.non_exited[side] >= self.K[side]:
            return
        other = 3 - side
        iid = self.next_id
        self.next_id += 1
        item = _SimItem(iid, side, self.t, status="new")
        self.items[iid] = item
        self.non_exited[side] += 1
        other_iid = self._select_waiting(other)
        if other_iid is not None:
            other_item = self.items[other_iid]
            other_item.status = "matched"
            item.status = "matched"
            wait = {1: 0.0, 2: 0.0}
            wait[side] = 0.0
            wait[other] = self.t - other_item.birth_time
            self._record_match(wait[1], wait[2])
            self._push(self.t + self.Tp + self.T[side], "exit", iid)
            self._push(self.t + self.Tp + self.T[other], "exit", other_iid)
        else:
            item.status = "waiting"
            self.waiting[side].append(iid)
            self._push(self.t + self.W[side], "expire", iid)


def _solve_sim(
    K1, K2, lam1, lam2, W1, W2, Tp, T1, T2, policy, *, horizon=500.0, warmup=50.0, replications=3, seed=12345
) -> dict[str, float]:
    vals = []
    for r in range(replications):
        sim = _DeterministicSimulator(K1, K2, lam1, lam2, W1, W2, Tp, T1, T2, policy, horizon, warmup, seed + 99991 * r)
        vals.append(sim.run())
    out: dict[str, float] = {}
    for key in ["expected_matching_rate", "expected_wait_time_EZ1", "expected_wait_time_EZ2"]:
        arr = np.array([v[key] for v in vals], dtype=float)
        out[key] = float(np.nanmean(arr))
        out[key + "_se"] = float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else math.nan
    return out


def solve_matching(K1, K2, lam1, lam2, W1, W2, Tp, T1, T2, policy, model):
    """
    Estimate matching rate and side-specific expected matched ages.

    Parameters
    ----------
    K1, K2 : int
        Capacities.
    lam1, lam2 : sequence of float
        State-dependent arrival rates. Length can be 1, K, or K+1.
    W1, W2 : float
        Deterministic cutoff windows
    Tp, T1, T2 : float
        Processing and delay times
    policy : str
        FIFO, LIFO, or RANDOM.
    model : str
        CTMC, BD, or SIM.

    Returns
    -------
    dict with expected_matching_rate, expected_wait_time_EZ1, expected_wait_time_EZ2.
    """
    policy = _normalize_policy(policy)
    model_u = model.strip().upper()
    l1 = _parse_rates(lam1, int(K1), "lam1")
    l2 = _parse_rates(lam2, int(K2), "lam2")
    K1, K2 = int(K1), int(K2)
    W1, W2 = float(W1), float(W2)
    if model_u.upper() in ["BD", "BDP", "SIMPLE"]:
        ans = _solve_bd(K1, K2, l1, l2, W1, W2, policy)
    elif model_u.upper() in ["CTMC", "FULL"]:
        ans = _solve_ctmc(K1, K2, l1, l2, W1, W2, float(Tp), float(T1), float(T2), policy)
    elif model_u.upper() == "SIM":
        ans = _solve_sim(K1, K2, l1, l2, W1, W2, float(Tp), float(T1), float(T2), policy)
    else:
        raise ValueError("model must be 'CTMC', 'BD', or 'SIM'")
    ans.update({"model": model_u, "policy": policy})
    return ans


type PauliError = tuple[float, Sequence[float]]
"""Channel fidelity and X,Y,Z error probabilities."""


class ComputedWaitTimeBudget(NamedTuple):
    F_req: float
    """Input fidelity requirement."""
    pauli: tuple[PauliError, PauliError]
    """Raw fidelity for each link."""
    T: tuple[float, float]
    """Interval between consecutive attempts.."""
    Tswp: float
    """Swapping time."""
    W: tuple[float, float]
    """Wait-time budgets for two links."""
    F_est: float
    """Estimated fidelity."""
    R_est: float
    """Estimated rate."""
    w2t: Callable[[Sequence[float] | None], float]
    """Wait-time to fidelity function."""


def compute_wait_time_budgets(
    *,
    F_req: float,
    L: Sequence[float],
    L_cohere: float = 250,
    lam: Sequence[float],
    gam: Sequence[float],
    T_proc: float = 10e-6,
    q: float = 0.5,
    depol: Sequence[float] = (0.03, 0.02),
    decohere_when_swap: bool = True,
) -> ComputedWaitTimeBudget:
    """
    Compute wait-time budgets.

    Args:
        F_req: Fidelity requirement.
        L: Length in kilometers for two links.
        L_cohere: Fiber decoherence length in kilometers.
        lam: Entanglement rates in Hz (i.e. etg/s) for two links.
        gam: Decoherence rates in Hz (inverse of t_cohere) for three nodes.
        T_proc: Local processing time in seconds.
        q: Swapping success probability.
        depol: Optical depolarization component.
        decohere_when_swap: Whether memory decoheres during swapping latency.
    """
    L_list = L
    lam1, lam2 = lam
    gam0, gam1, gam2 = gam
    optic_depol_1, optic_depol_2 = depol
    F = F_req

    # Links
    # L_list = [32, 18]
    # L_cohere = 250
    L_q1 = L_list[0]
    L_q2 = L_list[1]

    # entanglement generation rates:
    # lam1, lam2 = 75.577, 244.935

    # Time Parameters
    # T_proc = 10e-6  # local processing time
    T1, T2 = (
        L_list[0] / 2e5 + T_proc,
        L_list[1] / 2e5 + T_proc,
    )  # 1-way latency (sec) for each queue, based on fiber length and speed of light in fiber

    tau1, tau2 = 1 * T1, 1 * T2  # From excitation to heralding, 4T for Barrett-Kok, 1T for Single-Heraled protocol

    # q = 0.5  # swapping success probability
    Tswp = 2 * max(T1, T2)  # swapping time, ignored for BK protocol, but 1-round trip for SINGLE_HERALDED.

    # Raw fidelity and errors are due to optics and BSM, do not count memory decoherence here
    # They should be estimated from experiments and Tomography
    # Link 1
    # optic_depol_1 = 0.03  # depolarization: channel excess (Raman) photon rate, detector dark counts.
    optic_depha_1 = 1 - np.exp(-L_q1 / 2 / L_cohere)  # dephasing component of optical errors + BSM visibility.

    f_raw1, raw_errors1 = channel2Pauli(optic_depol_1, optic_depha_1)

    # Link 2
    # optic_depol_2 = 0.02  # depolarization: channel excess (Raman) photon rate, detector dark counts.
    optic_depha_2 = 1 - np.exp(-L_q2 / 2 / L_cohere)  # dephasing component of optical errors + BSM visibility.

    f_raw2, raw_errors2 = channel2Pauli(optic_depol_2, optic_depha_2)

    # Combine depolarization and dephasing errors
    v1, w1 = 1 - optic_depol_1, 1 - optic_depha_1
    v2, w2 = 1 - optic_depol_2, 1 - optic_depha_2

    swp_degradation = 1
    v_swp = v1 * v2 * swp_degradation  # degradation due to memory-memory swapping, can be estimated from experiments.
    w_swp = w1 * w2  # optical dephasing component of swapping error

    # Memory decoherence parameters: Assuming time-dependent noise.
    # These can be estimated from memory characterization experiments.
    # gam0 = 5  # decoherence rate for Node 0 (Hz)
    # gam1 = 10  # decoherence rate for Node 1 (Hz)
    # gam2 = 10  # decoherence rate for Node 2 (Hz)
    gam01, gam12 = gam0 + gam1, gam1 + gam2
    A0 = gam0 * (T1 + Tswp) + gam2 * (T2 + Tswp) + gam01 * tau1 + gam12 * tau2  # Minimum "Age", weighted by decoherence rates
    A0 += int(decohere_when_swap) * gam1 * 2 * Tswp

    def waittime2fidelity(x):
        return (1 + v_swp * (1 + 2 * w_swp * np.exp(-(A0 if x is None else A0 + gam01 * x[0] + gam12 * x[1])))) / 4

    # Varying the required fidelity for DEPHASING memory
    # Compute cutoff times and estimated Rate and Fidelity
    A_thr = -np.log(((4 * F - 1) / v_swp - 1) / (2 * w_swp))  # threshold for dephasing
    # optimal cutoff times for memories at node 1
    W1, W2 = (A_thr - A0) / gam01, (A_thr - A0) / gam12

    R = q * simple_rate(lam1, T1, W1, lam2, T2, W2, Tswp)
    wt = waittimes2swap([lam1, lam2], [W1, W2])
    F_est = waittime2fidelity(wt)

    return ComputedWaitTimeBudget(
        F_req=F,
        pauli=((f_raw1, raw_errors1), (f_raw2, raw_errors2)),
        T=(T1, T2),
        Tswp=Tswp,
        W=(W1, W2),
        F_est=F_est,
        R_est=R,
        w2t=waittime2fidelity,
    )
