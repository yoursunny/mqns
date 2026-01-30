from abc import ABC, abstractmethod
from typing import Protocol, TypedDict, Unpack, override

from mqns.entity.node import QNode
from mqns.models.epr import Entanglement
from mqns.simulator import Time


def _calc_propagation_loss(length: float, alpha: float) -> float:
    """
    Compute fiber propagation loss.

    Args:
        length: fiber length in kilometers.
        alpha: fiber loss in dB/km.

    Returns:
        Probability of a single photon to propagate through the fiber without loss.
    """
    return 10 ** (-alpha * length / 10)


class LinkArchParameters(TypedDict):
    length: float
    """Fiber length in kilometers."""
    alpha: float
    """Fiber loss in dB/km."""
    eta_s: float
    """Source efficiency between 0 and 1."""
    eta_d: float
    """Detector efficiency between 0 and 1."""
    reset_time: float
    """Inverse of source frequency in Hz."""
    tau_l: float
    """
    Fiber propagation delay in seconds.
    This is also used as one-way classical message delay.
    """
    tau_0: float
    """Local operation delay in seconds."""
    epr_type: type[Entanglement]
    """EPR type, either ``WernerStateEntanglement`` or ``MixedStateEntanglement``."""
    init_fidelity: float
    """Initial fidelity value."""


class LinkArch(Protocol):
    """
    Link architecture models the elementary entanglement generation protocol.

    Together with quantum channel and node hardware parameters, it supplies information to
    the skip-ahead sampling implementation in ``LinkLayer`` application.
    """

    name: str
    """Link architecture name."""

    success_prob: float
    """
    Success probability of a single attempt.

    This is available after `set()`.
    """

    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        """
        Save parameters about quantum channel and node hardware.
        """

    def delays(self, k: int) -> tuple[float, float, float]:
        """
        Compute protocol delays for k-th attempt.
        This is available after `set()`.

        Args:
            k: number of attempts, minimum is 1.

        Returns:
            Each value is a duration in seconds.

            * [0]: EPR creation time, since RESERVE_QUBIT_OK arrives at primary node.
            * [1]: notification time to primary node, since EPR creation.
            * [2]: notification time to secondary node, since EPR creation.
        """
        ...

    def make_epr(self, k: int, now: Time, *, key: str | None, src: QNode, dst: QNode) -> tuple[Entanglement, Time, Time]:
        """
        Create an elementary entanglement for k-th attempt.
        This is available after `set()`.

        Args:
            k: number of attempts, minimum is 1.
            now: current time point, which is when RESERVE_QUBIT_OK arrives at primary node.
            key: LinkLayer reservation key.
            src: primary node.
            dst: secondary node.

        Returns:
            Each value is a duration in seconds.

            * [0]: EPR object with fidelity assigned.
            * [1]: notification time point to primary node.
            * [2]: notification time point to secondary node.
        """
        ...


class LinkArchBase(ABC, LinkArch):
    def __init__(self, name: str):
        self.name = name
        self.success_prob = 0.0
        self.attempt_interval = 0.0
        self.d_notify_a = 0.0
        self.d_notify_b = 0.0

    @override
    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        self.success_prob = self._compute_success_prob(
            length=kwargs["length"],
            alpha=kwargs["alpha"],
            eta_s=kwargs["eta_s"],
            eta_d=kwargs["eta_d"],
        )

        self.attempt_interval, self.d_notify_a, self.d_notify_b = self._compute_delays(
            reset_time=kwargs["reset_time"],
            tau_l=kwargs["tau_l"],
            tau_0=kwargs["tau_0"],
        )

        self.epr_type = kwargs["epr_type"]
        self.init_fidelity = kwargs["init_fidelity"]

    @abstractmethod
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        """
        Compute success probability of a single attempt.
        Subclass implementation may precompute or save other parameters if necessary.
        """

    @abstractmethod
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        """
        Compute attempt interval and notification delays, for protocol delay computation.
        Subclass implementation may precompute or save other parameters if necessary.
        Override ``delays()`` method for unusual situations.
        """

    @override
    def delays(self, k: int) -> tuple[float, float, float]:
        return (k - 1) * self.attempt_interval, self.d_notify_a, self.d_notify_b

    @override
    def make_epr(self, k: int, now: Time, *, key: str | None, src: QNode, dst: QNode) -> tuple[Entanglement, Time, Time]:
        d_epr_creation, d_notify_a, d_notify_b = self.delays(k)
        t_epr_creation = now + d_epr_creation
        t_notify_a = now + (d_epr_creation + d_notify_a)
        t_notify_b = now + (d_epr_creation + d_notify_b)

        mem_a, mem_b = src.memory, dst.memory
        epr = self.epr_type(
            decohere_time=t_epr_creation + min(mem_a.decoherence_delay, mem_b.decoherence_delay),
            fidelity_time=t_epr_creation,
            src=src,
            dst=dst,
            store_decays=(mem_a.time_decay, mem_b.time_decay),
        )
        epr.fidelity = self.init_fidelity
        epr.key = key

        return epr, t_notify_a, t_notify_b


class LinkArchDimBk(LinkArchBase):
    """
    Detection-in-Midpoint link architecture with single-rail encoding using Barrett-Kok protocol.
    """

    def __init__(self, name="DIM-BK"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # Barrett-Kok uses single-rail encoding where the presence/absence of a photon indicates quantum state.
        # For a successful attempt, exactly one photon should arrive at the Bell-state analyzer M in each round.
        #
        # eta_sb is the probability of a photon triggering a detector, which consists of:
        # - eta_s: the source at A or B emits a photon.
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at M detects the photon.
        #
        # p_bsa, set to 50%, is the maximum theoretical coincidence probability for distinguishing two of
        # the four Bell states at a standard linear optics Bell-state analyzer.
        p_bsa = 0.5
        p_l_sb = _calc_propagation_loss(length / 2, alpha)
        eta_sb = eta_s * eta_d * p_l_sb
        return p_bsa * eta_sb**2

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. Both A and B emit photons at +τ0, which arrive at the Bell-state analyzer M at +(1/2)τl+τ0
        # 3. M sends heralding results to both A and B, which arrive at +τl+τ0
        # 4. Both A and B flip qubit gates locally
        # 5. Both A and B emit photons at +τl+2τ0, which arrive at M at +(1 1/2)τl+2τ0
        # 6. M sends heralding results to both A and B, which arrive at +2τl+2τ0
        # If either heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by twice of reset_time for two memory excitations.
        attempt_duration = 2 * (tau_l + tau_0)
        attempt_interval = max(attempt_duration, 2 * reset_time)
        return attempt_interval, attempt_duration, attempt_duration


class LinkArchDimBkSeq(LinkArchDimBk):
    """
    Detection-in-Midpoint link architecture with single-rail encoding using Barrett-Kok protocol,
    timing adjusted as per negotiation logic implemented by SeQUeNCe simulator.
    """

    def __init__(self, name="DIM-BK-SeQUeNCe"):
        super().__init__(name)

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # According to SeQUeNCe logic:
        # 1. A and B perform the first round negotiation, which completes at +2τl
        # 2. Qubits are generated at +2τl
        # 3. Both A and B emit photons at +2τl+τ0, which arrive at M at +(2 1/2)τl+τ0
        # 4. M sends heralding results to both A and B, which arrive at +3τl+τ0
        # 5. If the heralding result indicates failure, the current attempt is aborted
        # 6. A and B perform the second round negotiation, which completes at +5τl+τ0
        # 7. Both A and B flip qubit gates locally
        # 8. Both A and B emit photons at +5τl+2τ0, which arrive at M at +(5 1/2)τl+2τ0
        # 9. M sends heralding results to both A and B, which arrive at +6τl+2τ0
        # In summary, success occurs at +6τl+2τ0, failure occurs at either +3τl+τ0 or +6τl+2τ0.
        #
        # This model does not differentiate the two failure possibilities.
        # The attempt duration is set to 5τl+2τ0, in between two possible failed attempt durations.
        # It is also lower bounded by twice of reset_time for two memory excitations.
        #
        # The first round negotiation in the initial attempt, which takes 2τl, overlaps the reservation logic
        # in LinkLayer, so that the initial attempt is 2τl shorter.
        # For simplicity, this shortening is applied on the final attempt instead of the initial attempt.
        # Thus, the final attempt succeeds at +4τl+2τ0 as calculated in d_notify.
        attempt_duration = 5 * tau_l + 2 * tau_0
        attempt_interval = max(attempt_duration, 2 * reset_time)
        d_notify = 4 * tau_l + 2 * tau_0
        return attempt_interval, d_notify, d_notify


class LinkArchDimDual(LinkArchBase):
    """
    Detection-in-Midpoint link architecture with dual-rail polarization encoding.
    """

    def __init__(self, name="DIM-dual"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # For a successful attempt, one photon from each of A and B should arrive at the Bell-state analyzer M.
        #
        # eta_sb is the probability of a photon triggering a detector, which consists of:
        # - eta_s: the source at A or B emits a photon.
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at M detects the photon.
        #
        # p_bsa, set to 50%, is the maximum theoretical coincidence probability for distinguishing two of
        # the four Bell states at a standard linear optics Bell-state analyzer.
        p_bsa = 0.5
        p_l_sb = _calc_propagation_loss(length / 2, alpha)
        eta_sb = eta_s * eta_d * p_l_sb
        return p_bsa * eta_sb**2

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. Both A and B emit photons at +τ0, which arrive at the Bell-state analyzer M at +(1/2)τl+τ0
        # 3. M sends heralding results to both A and B, which arrive at +τl+τ0
        # If the heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one memory excitation.
        attempt_duration = tau_l + tau_0
        attempt_interval = max(attempt_duration, reset_time)
        return attempt_interval, attempt_duration, attempt_duration


class LinkArchSr(LinkArchBase):
    """
    Sender-Receiver link architecture with dual-rail polarization encoding.
    """

    def __init__(self, name="SR"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # The success probability consists of:
        # - eta_s: the source at B emits a photon.
        # - p_l_sr: the photon propagates through the fiber without loss.
        # - eta_d: the detector at A detects the photon.
        p_l_sr = _calc_propagation_loss(length, alpha)
        eta_sr = eta_s * eta_d * p_l_sr
        return eta_sr

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Qubits are generated at +0
        # 2. B emits a photon at +τ0, which arrives at +τl+τ0
        # 3. A absorbs the photon at +τl+τ0
        # 4. A sends heralding result to B, which arrives at +2τl+τ0
        # If the heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one memory excitation.
        attempt_duration = 2 * tau_l + tau_0
        attempt_interval = max(attempt_duration, reset_time)
        return attempt_interval, tau_l + tau_0, attempt_duration


class LinkArchSim(LinkArchBase):
    """
    Source-in-Midpoint link architecture with dual-rail polarization encoding.
    """

    def __init__(self, name="SIM"):
        super().__init__(name)

    @override
    def _compute_success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        # For a successful attempt, A and B must each receive a photon from the pair.
        #
        # eta_d*p_l_sb is the probability of a photon reaching either A or B, which consists of:
        # - p_l_sb: the photon propagates through the fiber without loss.
        # - eta_d: the detector at A or B detects the photon.
        #
        # The overall success probability has `**2` because it requires both photons.
        _ = eta_s
        p_l_sb = _calc_propagation_loss(length / 2, alpha)
        eta_rr = (eta_d * p_l_sb) ** 2
        return eta_rr

    @override
    def _compute_delays(self, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        # Reservation and setup were completed by LinkLayer.
        # In each attempt:
        # 1. Entangled photon pairs are continuously emitted by the entangled photon source
        # 2. Both A and B prepare their local qubits at +0, which completes at +τ0
        # 3. A absorbs a photon at +τ0; B absorbs the paired photon at +τ0
        # 4. A sends heralding result to B, which arrives at +τl+τ0; B does the same in opposite direction
        # If either heralding result indicates failure, the next attempt can immediately start.
        # The attempt interval is lower bounded by reset_time for one local qubit preparation.
        attempt_duration = tau_l + tau_0
        attempt_interval = max(attempt_duration, reset_time)
        return attempt_interval, attempt_duration, attempt_duration


class LinkArchAlways(LinkArch):
    """
    Link architecture wrapper that always succeeds, primarily for unit testing.
    """

    def __init__(self, inner: LinkArch):
        self.name = f"{inner.name}-always"
        self.inner = inner

    @override
    def set(self, **kwargs: Unpack[LinkArchParameters]) -> None:
        self.inner.set(**kwargs)
        self.success_prob = 1.0

    @override
    def delays(self, k: int) -> tuple[float, float, float]:
        assert k == 1
        return self.inner.delays(k)

    @override
    def make_epr(self, k: int, *args, **kwargs):
        assert k == 1
        return self.inner.make_epr(k, *args, **kwargs)
