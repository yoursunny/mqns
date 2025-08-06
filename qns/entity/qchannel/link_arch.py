from abc import ABC, abstractmethod

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class LinkArch(ABC):
    """Link architecture."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        """
        Compute success probability of a single attempt.

        Args:
            length: fiber length in kilometers.
            alpha: fiber loss in dB/km.
            eta_s: source efficiency between 0 and 1.
            eta_d: detector efficienccy between 0 and 1
        """
        pass

    @abstractmethod
    def delays(self, k: int, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        """
        Compute average duration of a single attempt.

        Args:
            k: number of attempts, minimum is 1.
            reset_time: inverse of source frequency in Hz.
            tau_l: fiber propagation delay.
            tau_0: local operation delay.

        Returns:
            [0]: when should k-th attempt succeed.
            [1]: EPR creation time before successful attempt.
            [2]: primary node notification time after successful attempt.
        """
        pass


class LinkArchDimBk(LinkArch):
    """
    Detection-in-Midpoint link architecture with Barrett-Kok protocol.
    """

    def __init__(self, name="DIM-BK"):
        super().__init__(name)

    def success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        p_bsa = 0.5
        p_l_sb = 10 ** (-alpha * length / 2 / 10)
        eta_sb = eta_s * eta_d * p_l_sb
        return p_bsa * eta_sb**2

    def delays(self, k: int, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        tau = 2 * (tau_l + tau_0)
        attempt_duration = max(tau, reset_time)
        return k * attempt_duration - tau_l, tau_0 + tau_l, tau_l


class LinkArchDimBkSeq(LinkArchDimBk):
    """
    Detection-in-Midpoint link architecture with Barrett-Kok protocol,
    with reservation logic from SeQUeNCe simulator.
    """

    def __init__(self, name="DIM-BK-SeQUeNCe"):
        super().__init__(name)

    @override
    def delays(self, k: int, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        tau = tau_l + tau_0
        attempt_duration = max(5 * tau, reset_time)
        return (k - 1) * attempt_duration + 5 * tau - tau_l, tau_0 + 3 * tau_l, tau_l


class LinkArchSr(LinkArch):
    """
    Sender-Receiver link architecture.
    """

    def __init__(self, name="SR"):
        super().__init__(name)

    def success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        p_l_sr = 10 ** (-alpha * length / 10)
        eta_sr = eta_s * eta_d * p_l_sr
        return eta_sr

    def delays(self, k: int, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        tau = 2 * (tau_l + tau_0)
        attempt_duration = max(tau, reset_time)
        return k * attempt_duration - tau_l, tau_l, 0


class LinkArchSim(LinkArch):
    """
    Source-in-Midpoint link architecture.
    """

    def __init__(self, name="SIM"):
        super().__init__(name)

    def success_prob(self, *, length: float, alpha: float, eta_s: float, eta_d: float) -> float:
        _ = eta_s
        p_l_sb = 10 ** (-alpha * length / 2 / 10)
        eta_rr = (eta_d * p_l_sb) ** 2
        return eta_rr

    def delays(self, k: int, *, reset_time: float, tau_l: float, tau_0: float) -> tuple[float, float, float]:
        tau = tau_l + tau_0
        attempt_duration = max(tau, reset_time)
        return k * attempt_duration - tau_l, 0, tau_l
