#    Multiverse Quantum Network Simulator: a simulator for comparative
#    evaluation of quantum routing strategies
#    Copyright (C) [2025] Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import deque
from collections.abc import Iterable
from typing import Final, Literal, TypedDict, final, override

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, classic_cmd_handler
from mqns.entity.memory import MemoryQubit, QubitState
from mqns.entity.node import Application, QNode
from mqns.entity.qchannel import QuantumChannel
from mqns.network.network import TimingPhase, sync_phase_handler
from mqns.network.protocol.event import (
    LinkArchNotify2ndEvent,
    LinkArchNotifyPriEvent,
    PathActivateEvent,
    PathDeactivateEvent,
    QubitEntangledEvent,
    QubitReleasedEvent,
)
from mqns.simulator import event_handler
from mqns.utils import AutoIncrementIdentifier, json_encodable, rng, unwrap

_AUTOID = AutoIncrementIdentifier("llk_")
"""
Automatically assigned ``ReservationRequest.key`` name.
"""


class ReserveMsg(TypedDict):
    """
    Message between LinkLayers to request and accept qubit reservation for entanglement generation.
    """

    cmd: Literal["RESERVE_REQ", "RESERVE_RES"]
    """
    RESERVE_REQ is sent from primary node to secondary node to request entanglement generation.
    RESERVE_RES is sent from secondary node to primary node to accept entanglement generation.
    """
    key: str
    """
    Reservation key that uniquely identifies this reservation and the resulting entanglement.
    """
    path_id: int | None
    """
    The path identifier assigned to the generated entanglement.
    """


@final
class _ActivePath:
    """
    LinkLayer data structure related to an active path in an active quantum channel.
    """

    __slots__ = ("path_id", "insertion_count", "oreq_table", "ireq_queue")

    path_id: Final[int | None]
    """
    Path identifier.

    LinkLayer considers ``None`` as a valid path_id (rather than an absent value), because the forwarder's
    multiplexing scheme may not need to allocate each qubit to an integer path_id.
    """

    insertion_count: int
    """
    How many times this channel+path_id combination has been activated.
    It would take the same number of deactivation commands to deactivate the channel+path_id combination.
    """

    oreq_table: dict[str, int]
    """
    Table of outgoing RESERVE_REQ sent by this node for which a reply has not arrived.
    This field only exists on the primary node of the channel.
    Key is reservation key.
    Value is qubit address.
    """

    ireq_queue: deque[str]
    """
    Queue of incoming RESERVE_REQ received by this node that has not been fulfilled.
    This field only exists on the secondary node of the channel.
    Element is reservation key.
    """

    def __init__(self, path_id: int | None, is_primary: bool):
        self.path_id = path_id
        self.insertion_count = 0
        if is_primary:
            self.oreq_table = {}
        else:
            self.ireq_queue = deque()


@final
class _ActiveChannel:
    """
    LinkLayer data structure related to an active quantum channel.
    """

    __slots__ = ("qchannel", "partner", "is_primary", "paths")

    qchannel: Final[QuantumChannel]
    """Quantum channel."""

    partner: Final[QNode]
    """Partner node."""

    is_primary: Final[bool]
    """Role of this node."""

    paths: dict[int | None, _ActivePath]
    """
    Active paths.
    Key is ``path_id``.
    Value is ActivePath record.
    """

    def __init__(self, qchannel: QuantumChannel, partner: QNode, is_primary: bool):
        self.qchannel = qchannel
        self.partner = partner
        self.is_primary = is_primary
        self.paths = {}


@json_encodable
class LinkLayerCounters:
    @staticmethod
    def aggregate(nodes: Iterable[QNode]) -> "LinkLayerCounters":
        """
        Aggregate ``LinkLayerCounters`` from a network.

        Args:
            nodes: List of nodes, such as ``QuantumNetwork.nodes``.
        """
        r = LinkLayerCounters()
        for node in nodes:
            for ll in node.get_apps(LinkLayer):
                r.n_etg += ll.cnt.n_etg
                r.n_attempts += ll.cnt.n_attempts
                r.n_decoh += ll.cnt.n_decoh
        return r

    def __init__(self):
        self.n_etg = 0
        """how many entanglements generated as the primary node"""
        self.n_attempts = 0
        """how many attempts made for successful entanglements"""
        self.n_decoh = 0
        """how many qubits decohered"""

    def increment_n_etg(self, attempts: int) -> None:
        self.n_etg += 1
        self.n_attempts += attempts

    @property
    def decoh_ratio(self) -> float:
        """decoherence ratio, ``n_decoh/n_etg``"""
        return self.n_decoh / self.n_etg if self.n_etg > 0 else 0

    def __repr__(self) -> str:
        return f"etg={self.n_etg} attempts={self.n_attempts} decoh={self.n_decoh} decoh_ratio={self.decoh_ratio}"


class LinkLayer(ClassicCommandDispatcherMixin, Application[QNode]):
    """
    Network function for creating elementary entanglements over qchannels.
    It equips a QNode and is activated from the forwarding function (e.g., ProactiveForwarder).
    """

    def __init__(
        self,
        *,
        attempt_rate: float = 1e6,
        eta_s: float = 1.0,
        eta_d: float = 1.0,
        frequency: float = 80e6,
        tau_0: float = 0.0,
    ):
        """
        Constructor.

        Args:
            attempt_rate: max entanglement attempts per second (default: 1e6) (currently ineffective).
            eta_s: source efficiency (default: 1.0).
            eta_d: detector efficiency (default: 1.0).
            frequency: entanglement source frequency in Hz (default: 80e6).
            tau_0: local operation delay in seconds for emitting and absorbing photon (default: 0.0).

        """
        super().__init__()

        self.attempt_interval = 1 / attempt_rate
        """Minimum interval spaced out between attempts (currently ineffective)."""
        self.eta_s = eta_s
        """Source efficiency between 0 and 1."""
        self.eta_d = eta_d
        """Detector efficiency between 0 and 1."""
        self.reset_time = 1 / frequency
        """Minimum time between two consecutive photon excitations/absorptions."""
        self.tau_0 = tau_0
        """Local operation delay in seconds."""

        self.channels: dict[str, _ActiveChannel] = {}
        """
        Active channel table.
        Key is partner node name.
        """

        self.cnt = LinkLayerCounters()
        """
        Counters.
        """

    @override
    def install(self, node):
        self._application_install(node, QNode)
        self.memory = self.node.memory
        """Quantum memory of the node."""

    @sync_phase_handler(TimingPhase.EXTERNAL, True)
    def sync_external_enter(self) -> None:
        """
        In SYNC timing mode, enter EXTERNAL phase.
        """
        # Start reservation for each active channel where this node is primary.
        for ac in self.channels.values():
            if ac.is_primary:
                self.run_channel(ac)

    @sync_phase_handler(TimingPhase.EXTERNAL, False)
    def sync_external_exit(self) -> None:
        """
        In SYNC timing mode, exit EXTERNAL phase.
        """
        for ac in self.channels.values():
            if ac.is_primary:
                # Clear incomplete reservations.
                for ap in ac.paths.values():
                    ap.oreq_table.clear()
            else:
                # Clear unsatisfied reservation requests.
                for ap in ac.paths.values():
                    ap.ireq_queue.clear()

    @sync_phase_handler(TimingPhase.INTERNAL, False)
    def sync_internal_exit(self) -> None:
        """
        In SYNC timing mode, exit INTERNAL phase.
        """
        # Clear existing memory qubits.
        self.memory.clear()

    @event_handler
    def path_activate(self, event: PathActivateEvent) -> None:
        """
        Handle path activation command.
        """
        event.cancel()
        ch = event.qchannel
        partner = ch.find_peer(self.node)
        path_id = event.path_id

        ac = self.channels.get(partner.name)
        if not ac:
            ac = _ActiveChannel(ch, partner, event.is_primary)
            self.channels[partner.name] = ac

            self._path_activate_channel(ac)

        ap = ac.paths.get(path_id)
        if not ap:
            ap = _ActivePath(path_id, ac.is_primary)
            ac.paths[path_id] = ap

            addrs = list(q.addr for q, _ in self.memory.find(lambda *_: True, qchannel=ch))
            self.log_debug(
                "PATH_ACTIVATE_%s %s path=%s partner=%s has-addrs=%s",
                PathActivateEvent.ROLE_STR[ac.is_primary],
                ch.name,
                path_id,
                partner.name,
                addrs,
            )

        ap.insertion_count += 1

        if ac.is_primary and self.node.timing.is_async():
            self.run_channel(ac, ap)

    def _path_activate_channel(self, ac: _ActiveChannel) -> None:
        ch = ac.qchannel
        if not ac.is_primary:
            self.log_debug("CHANNEL_ACTIVATE_2nd %s partner=%s", ch.name, ac.partner.name)
            return

        # link_arch.set() may be called multiple times in several situations:
        # - qchannel is activated, deactivated, and re-activated.
        # - qchannel is activated from both sides due to shared paths in opposite directions.
        # Nevertheless, we assume every LinkLayer instance has the same parameters,
        # so that the parameters saved into LinkArch are the same every time.
        ch.link_arch.set(
            ch=ch,
            eta_s=self.eta_s,
            eta_d=self.eta_d,
            reset_time=self.reset_time,
            tau_0=self.tau_0,
            epr_type=self.node.network.epr_type,
            t0=self.simulator.tc,
            store_decays=(self.memory.time_decay, ac.partner.memory.time_decay),
        )

        epr_tpl, t_notify_a, t_notify_b = ch.link_arch.make_epr(1, self.simulator.ts, src=self.node, dst=ac.partner, key=None)

        self.log_debug(
            "CHANNEL_ACTIVATE_pri %s partner=%s link_arch=%s t_notify=%s,%s | template %s",
            ch.name,
            ac.partner.name,
            ch.link_arch.name,
            t_notify_a,
            t_notify_b,
            epr_tpl,
        )

    @event_handler
    def path_deactivate(self, event: PathDeactivateEvent) -> None:
        """
        Handle path deactivation command.
        """
        event.cancel()
        ch = event.qchannel
        partner = ch.find_peer(self.node)
        path_id = event.path_id

        ac = self.channels[partner.name]
        assert ac.qchannel is ch

        ap = ac.paths[path_id]
        ap.insertion_count -= 1
        if ap.insertion_count > 0:
            return

        # If the path on a channel is being deactivated, reset qubits owned by LinkLayer.
        resets: list[tuple[int, str | None]] = []
        for mq, _ in self.memory.find(
            lambda q, _: q.state in (QubitState.ACTIVE, QubitState.RESERVED) and q.path_id == path_id, qchannel=ch
        ):
            resets.append((mq.addr, mq.key))
            mq.state = QubitState.RAW

        del ac.paths[path_id]
        self.log_debug(
            "PATH_DEACTIVATE_%s %s path=%s partner=%s reset-qubits=%s",
            PathActivateEvent.ROLE_STR[ac.is_primary],
            ch.name,
            path_id,
            partner.name,
            resets,
        )
        if len(ac.paths) > 0:
            return

        del self.channels[partner.name]
        self.log_debug("CHANNEL_DEACTIVATE_%s %s partner=%s", PathActivateEvent.ROLE_STR[ac.is_primary], ch.name, partner.name)

    def run_channel(self, ac: _ActiveChannel, ap: _ActivePath | None = None) -> None:
        """
        Start entanglement generation on qubits assigned to a channel.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            ap: If set, only consider qubits allocated to a specified path.
        """
        qubits = self.memory.find(
            (lambda q, _: q.state is QubitState.RAW)
            if ap is None
            else (lambda q, _: q.state is QubitState.RAW and q.path_id == ap.path_id),
            qchannel=ac.qchannel,
        )
        for q, v in qubits:
            assert q.key is None
            assert v is None
            self.start_reservation(ac, q)

    def start_reservation(self, ac: _ActiveChannel, mq: MemoryQubit):
        """
        Start entanglement generation on a qubit assigned to a channel.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            mq: The qubit, which must be in RAW state.
        """
        assert ac.is_primary

        # Generate a unique reservation key to represent this entanglement generation protocol execution
        # and the entanglement, which is valid until the entanglement is released.
        mq.key = _AUTOID()
        mq.state = QubitState.ACTIVE

        # Remember the pending reservation in ActivePath record.
        ap = ac.paths[mq.path_id]
        ap.oreq_table[mq.key] = mq.addr
        self.log_debug(
            "RESERVE_REQ(%s:%s, %s) sent addr=%s secondary=%s", self.node.name, mq.path_id, mq.key, mq.addr, ac.partner.name
        )

        # Transmit RESERVE_REQ message to the secondary node.
        msg = ReserveMsg(cmd="RESERVE_REQ", key=mq.key, path_id=mq.path_id)
        self.node.send_cpacket(ac.partner, ClassicPacket(msg, src=self.node, dest=ac.partner))

    @classic_cmd_handler("RESERVE_REQ")
    def handle_reserve_req(self, pkt: ClassicPacket, msg: ReserveMsg):
        """
        Handle ``RESERVE_REQ`` message from the primary node.
        """
        key = msg["key"]
        path_id = msg["path_id"]

        if not self.node.timing.is_external():
            self.log_debug("RESERVE_REQ(%s:%s, %s) ignored reason=not-external-phase", pkt.src.name, path_id, key)
            return

        ac = self.channels.get(pkt.src.name)
        if ac is None:
            self.log_debug("RESERVE_REQ(%s:%s, %s) ignored reason=no-active-channel", pkt.src.name, path_id, key)
            return
        assert not ac.is_primary

        ap = ac.paths.get(path_id)
        if ap is None:
            self.log_debug("RESERVE_REQ(%s:%s, %s) ignored reason=no-active-path", pkt.src.name, path_id, key)
            return

        if len(ap.ireq_queue) > 0 or not self.try_accept_reservation(ac, ap, key):
            # If the queue is non-empty, try_accept_reservation() would not have succeeded.
            # If try_accept_reservation() was unsuccessful, enqueue the request.
            ap.ireq_queue.append(key)

    def try_accept_reservation(self, ac: _ActiveChannel, ap: _ActivePath, key: str, *, hint: MemoryQubit | None = None) -> bool:
        """
        Accept a reservation if a qubit is available.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is secondary.
            ap: ActivePath record of the path to which the qubit is allocated.
            key: Reservation key from a received RESERVE_RES message.
            hint: If provided, it will be used, bypassing the search; it must be in RAW state.

        Returns:

            * True if the request is accepted and ``RESERVE_RES`` is sent.
              Caller is responsible for dequeuing the request if it was in a queue.
            * False if the request is not accepted.
              Caller is responsible for enqueuing the request.
        """
        assert not ac.is_primary

        if hint:
            mq = hint
            assert mq.state is QubitState.RAW
            assert mq.path_id == ap.path_id
        else:
            mq, _ = next(
                self.memory.find(
                    lambda q, _: (
                        q.state is QubitState.RAW  # currently unoccupied and not part of an active reservation
                        and q.path_id == ap.path_id  # allocated to the path_id, if MuxScheme uses path_id
                    ),
                    qchannel=ac.qchannel,  # assigned to the quantum channel
                ),
                (None, None),
            )

        if mq is None:
            return False

        self.log_debug("RESERVE_REQ(%s:%s, %s) accepted addr=%s", ac.partner.name, ap.path_id, key, mq.addr)
        mq.state = QubitState.RESERVED
        mq.key = key
        msg = ReserveMsg(cmd="RESERVE_RES", key=key, path_id=ap.path_id)
        self.node.send_cpacket(ac.partner, ClassicPacket(msg, src=self.node, dest=ac.partner))
        return True

    @classic_cmd_handler("RESERVE_RES")
    def handle_reserve_res(self, pkt: ClassicPacket, msg: ReserveMsg):
        """
        Handle ``RESERVE_RES`` message from the secondary node.
        """
        key = msg["key"]
        path_id = msg["path_id"]

        if not self.node.timing.is_external():
            self.log_debug("RESERVE_RES(%s:%s, %s) ignored reason=not-external-phase", pkt.src.name, path_id, key)
            return

        ac = self.channels.get(pkt.src.name)
        if ac is None:
            self.log_debug("RESERVE_RES(%s:%s, %s) ignored reason=no-active-channel", pkt.src.name, path_id, key)
            return

        ap = ac.paths.get(path_id)
        if ap is None:
            self.log_debug("RESERVE_RES(%s:%s, %s) ignored reason=no-active-path", pkt.src.name, path_id, key)
            return

        addr = ap.oreq_table.pop(key, None)
        if addr is None:
            self.log_debug("RESERVE_RES(%s:%s, %s) ignored reason=no-active-key", pkt.src.name, path_id, key)
            return

        mq, _ = self.memory.read(addr, must=True)
        assert mq.key == key
        mq.state = QubitState.RESERVED
        self.log_debug("RESERVE_RES(%s:%s, %s) processed addr=%s", pkt.src.name, path_id, key, mq.addr)
        self.generate_entanglement(ac, mq)

    def generate_entanglement(self, ac: _ActiveChannel, mq: MemoryQubit) -> None:
        """
        Schedule a successful entanglement attempt using skip-ahead sampling.

        Args:
            ac: ActiveChannel record of the quantum channel, where this node is primary.
            qubit: The memory qubit used for this attempt.
        """
        qchannel = ac.qchannel
        src = self.node
        dst = ac.partner
        key = unwrap(mq.key)

        # Calculate which attempt would succeed.
        k = self._calc_attempt(qchannel)

        # Calculate when would the k-th attempt (1-based) succeed.
        # TODO space out EPRs on a qchannel by attempt_interval or qchannel.bandwidth
        epr, t_notify_a, t_notify_b = qchannel.link_arch.make_epr(k, self.simulator.tc, src=src, dst=dst, key=key)

        # If the network uses SYNC timing mode but the successful attempt would exceed the current EXTERNAL phase,
        # the EPR would not arrive in time, and therefore is not scheduled.
        if not src.timing.is_external(max(t_notify_a, t_notify_b)):
            self.log_debug(
                "EPR_SKIP %s dst=%s key=%s attempts=%s notify-times=%s,%s reason=beyond-external-phase",
                epr.name,
                dst.name,
                key,
                k,
                t_notify_a,
                t_notify_b,
            )
            return

        # If the network uses ASYNC timing mode or the successful attempt can complete within the current EXTERNAL phase,
        # schedule the EPR arrival on both nodes via LinkArchSuccessEvents.
        self.log_debug(
            "EPR_PREPARE %s dst=%s key=%s attempts=%s notify-times=%s,%s", epr.name, dst.name, key, k, t_notify_a, t_notify_b
        )

        self.simulator.add_event(LinkArchNotifyPriEvent(key, epr, t=t_notify_a, attempts=k))
        self.simulator.add_event(LinkArchNotify2ndEvent(key, epr, t=t_notify_b))

    def _calc_attempt(self, qchannel: QuantumChannel) -> int:
        return rng.geometric(qchannel.link_arch.success_prob)

    @event_handler
    def _la_notify_pri(self, event: LinkArchNotifyPriEvent) -> None:
        self.cnt.increment_n_etg(event.attempts)
        self._la_notify("pri", event)

    @event_handler
    def _la_notify_2nd(self, event: LinkArchNotify2ndEvent) -> None:
        self._la_notify("2nd", event)

    def _la_notify(self, own_role: str, event: LinkArchNotifyPriEvent | LinkArchNotify2ndEvent) -> None:
        event.cancel()
        assert self.node.timing.is_external()
        partner = event.partner
        key = event.key
        epr = event.epr
        try:
            mq = self.memory.write(key, epr)
        except LookupError:
            # Path was deactivated and qubit was deallocated.
            self.log_debug(
                "EPR_SKIP_%s %s partner=%s key=%s reason=qubit-not-found",
                own_role,
                epr.name,
                partner.name,
                key,
            )
            return

        self.log_debug(
            "EPR_DELIVER_%s %s partner=%s key=%s addr=%s path=%s",
            own_role,
            epr.name,
            partner.name,
            key,
            mq.addr,
            mq.path_id,
        )
        assert epr.decohere_time > self.simulator.tc

        mq.state = QubitState.ENTANGLED0
        self.simulator.add_event(QubitEntangledEvent(self.node, partner, mq, t=self.simulator.tc))

    @event_handler
    def handle_release(self, event: QubitReleasedEvent) -> None:
        mq = event.qubit
        mq.state = QubitState.RAW

        partner = unwrap(mq.qchannel).find_peer(self.node)

        ac = self.channels.get(partner.name)
        if ac is None:
            self.log_debug("%s ignored reason=no-active-channel", event)
            return

        ap = ac.paths.get(mq.path_id)
        if ap is None:
            self.log_debug("%s ignored reason=no-active-path", event)
            return

        if ac.is_primary:
            self.log_debug("%s processed role=primary", event)
            if event.is_decoh:
                self.cnt.n_decoh += 1
            if self.node.timing.is_async():
                self.start_reservation(ac, mq)
        else:
            self.log_debug("%s processed role=secondary", event)
            if ap.ireq_queue and self.try_accept_reservation(ac, ap, ap.ireq_queue[0], hint=mq):
                ap.ireq_queue.popleft()
