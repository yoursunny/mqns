from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from mqns.entity.memory import MemoryQubit, QuantumMemory, QubitState
from mqns.entity.node import QNode
from mqns.models.delay import DelayModel
from mqns.models.epr import Entanglement
from mqns.models.error import ErrorModel
from mqns.network.fw.fib import FibEntry
from mqns.network.fw.message import SwapUpdateMsg
from mqns.network.fw.mux import MuxScheme
from mqns.network.network import QuantumNetwork
from mqns.simulator import Simulator, Time, func_to_event
from mqns.utils import log

if TYPE_CHECKING:
    from mqns.network.fw.forwarder import Forwarder


_OUTCOME_STR = {True: "SUCCESS", False: "FAILURE"}


@dataclass
class PartnerInfo:
    """Items related to either prev(left) or next(right) partner."""

    partner: QNode
    """Partner node."""
    partner_index: int
    """Partner node index within FIB entry."""
    partner_rank: int
    """Partner node swap rank within FIB entry."""
    partner_attr: Literal["src", "dst"]
    """``Entanglement`` attribute to retrieve partner."""


@dataclass
class SwapArm(PartnerInfo):
    """Items related to one arm of a swap operation."""

    addr: int
    """Local qubit address."""
    epr: Entanglement
    """Logical EPR with partner."""
    phy_epr: Entanglement
    """Physical EPR toward partner."""


@dataclass
class SwapContext:
    """Items related to a swap operation."""

    fib_entry: FibEntry
    """FIB entry."""
    arms: tuple[SwapArm, SwapArm]
    """Prev and next arms."""
    local_success: bool
    """Whether the local BSA was successful."""
    phy_outcome: Entanglement
    """Physical swap outcome."""


@dataclass
class HeraldContext(PartnerInfo):
    """Items related to received heralding."""

    fib_entry: FibEntry
    """FIB entry."""

    old_epr_name: str
    """Logical EPR name with old partner."""
    heralded_success: bool
    """Whether all swaps between self node and partner were successful according to heralding."""
    heralded_expiry: Time
    """Earliest known qubit expiration time."""


class ForwarderSwapProc:
    """
    Part of ``Forwarder`` logic related to swapping procedure.
    """

    fw: "Forwarder"
    simulator: Simulator
    epr_type: type[Entanglement]
    network: QuantumNetwork
    node: QNode
    memory: QuantumMemory
    mux: MuxScheme

    def __init__(self, *, ps: float, delay: DelayModel, error: ErrorModel):
        self.ps = ps
        """Probability of successful entanglement swapping."""
        assert 0.0 <= self.ps <= 1.0
        self.delay = delay
        """Swapping delay model."""
        self.error = error
        """Swapping error model."""

        self.waiting_su: dict[int, tuple[SwapUpdateMsg, FibEntry]] = {}
        """
        SwapUpdates received prior to QubitEntangledEvent.

        * Key: MemoryQubit addr.
        * Value: SwapUpdateMsg and FibEntry.
        """

        self.heralded_expiry: dict[str, Time] = {}

        self.local_swapped: dict[str, tuple[Entanglement, bool]] = {}

        self.remote_swapped: dict[str, Entanglement] = {}
        """
        EPRs that have been swapped remotely but the SwapUpdateMsg have not arrived.
        Each key is an EPR name; each value is the EPR.

        When a remote forwarder performs a swapping in which this node is either src or dst of the new EPR,
        it deposits the swapped EPR here and transmits the corresponding SwapUpdateMsg.
        Upon receiving the SwapUpdateMsg, the local forwarder pops the EPR.

        XXX Current approach assumes cchannels do not have packet loss.
        """

    def install(self, fw: "Forwarder"):
        self.fw = fw
        self.simulator = fw.simulator
        self.epr_type = fw.epr_type
        self.network = fw.network
        self.node = fw.node
        self.memory = fw.memory
        self.mux = fw.mux

    def _deposit_remote_swapped(self, target: QNode, old_epr_name: str, new_epr: Entanglement):
        target.get_app(type(self.fw)).swap.remote_swapped[old_epr_name] = new_epr

    def _cleanup(self, old_epr_name: str):
        self.heralded_expiry.pop(old_epr_name, None)
        self.local_swapped.pop(old_epr_name, None)
        self.remote_swapped.pop(old_epr_name, None)

    def _herald_to(self, fib_entry: FibEntry, target: PartnerInfo, opposite: PartnerInfo, old_epr_name: str, expiry: int):
        own_rank = fib_entry.own_swap_rank
        if target.partner_rank < own_rank:
            raise RuntimeError(f"{self.node} cannot herald to lower-ranked {target.partner}")
        if target.partner_rank > own_rank and opposite.partner_rank <= own_rank:
            # Self node is still part of a lower-ranked parallel group, no ready to herald target.
            return

        su_msg: SwapUpdateMsg = {
            "cmd": "SWAP_UPDATE",
            "path_id": fib_entry.path_id,
            "swapping_node": self.node.name,
            "old_epr": old_epr_name,
            "partner": opposite.partner.name,
            "expiry": expiry,
        }
        self.fw.send_msg(target.partner, su_msg, fib_entry)

    def start(self, mq0: MemoryQubit, mq1: MemoryQubit, fib_entry: FibEntry):
        """
        Start swapping between two qubits at an intermediate node.

        Args:
            mq0: First qubit, must be in ELIGIBLE state.
            mq1: Second qubit, must be in ELIGIBLE state and come from a different qchannel.
            fib_entry: FIB entry.
        """
        assert mq0.addr != mq1.addr

        # Read both qubits and remove them from memory.
        #
        # One of these qubits must be entangled with a partner node to the left of the current node.
        # This is determined by epr.dst==self.node condition, because LinkLayer establishes elementary
        # entanglements from left to right, and swapping maintains this condition.
        # This qubit and related objects are assigned to `prev`.
        #
        # Likewise, the other qubit entangled with a partner node to the right is assigned to `next`.
        prev = self._retrieve_arm(mq0, fib_entry)
        next = self._retrieve_arm(mq1, fib_entry)
        if prev.partner_index > next.partner_index:
            next, prev = prev, next

        # Perform the physical swap.
        now = self.simulator.tc
        new_phy_epr, local_success = Entanglement.swap(prev.phy_epr, next.phy_epr, now=now, ps=self.ps, error=self.error)

        # Make deposits of the physical outcome.
        for target, opposite in (prev, next), (next, prev):
            if target.partner_rank <= fib_entry.own_swap_rank:
                # If target.partner would send us SwapUpdate, record local tracking,
                # which allows self node to match incoming heralds to physical outcomes.
                # It would be popped in `_su_parallel` or `_su_sequential`.
                self.local_swapped[target.epr.name] = new_phy_epr, local_success

                # If target.partner is same-rank, they need the physical EPR for their parallel swap.
                # It would be popped in `_su_parallel` or `_retrieve_arm`.
                if target.partner_rank == fib_entry.own_swap_rank:
                    self._deposit_remote_swapped(target.partner, target.epr.name, new_phy_epr)

            # If target.partner is higher-ranked, they need the physical EPR for sequential extension,
            # but only if the parallel group can possibly herald them.
            # It would be popped in `_su_sequential`.
            elif opposite.partner_rank > fib_entry.own_swap_rank:
                self._deposit_remote_swapped(target.partner, target.epr.name, new_phy_epr)

        # Schedule swap completion.
        # TODO skip callback and cleanup when using SYNC timing and t would be exceed INTERNAL phase.
        t = now + self.delay.calculate()
        ctx = SwapContext(fib_entry, (prev, next), local_success, new_phy_epr)
        self.simulator.add_event(func_to_event(t, self._after_swap, (mq0, mq1), ctx))

    def _retrieve_arm(self, mq: MemoryQubit, fib_entry: FibEntry) -> SwapArm:
        assert mq.state is QubitState.ELIGIBLE
        _, epr = self.memory.read(mq.addr, has=self.epr_type, remove=True)

        if epr.dst is self.node:
            partner, partner_attr, ch_index_offset = epr.src, "src", -1
        elif epr.src is self.node:
            partner, partner_attr, ch_index_offset = epr.dst, "dst", 0
        else:
            raise ValueError(f"{self.node} is neither src nor dst of {epr}")
        assert partner is not None
        index, rank = fib_entry.find_index_and_swap_rank(partner.name)

        if not epr.orig_eprs:
            epr.ch_index = fib_entry.own_idx + ch_index_offset

        return SwapArm(
            partner=partner,
            partner_index=index,
            partner_rank=rank,
            partner_attr=partner_attr,
            addr=mq.addr,
            epr=epr,
            phy_epr=self.remote_swapped.pop(epr.name, epr),
        )

    def _after_swap(self, qubits: Iterable[MemoryQubit], ctx: SwapContext):
        """
        Bell-State Analyzer completion event handler.

        Args:
            qubits: Qubits consumed by local swapping.
            ctx: Swapping operations parameters and results.
        """
        # Release old qubits.
        for mq in qubits:
            self.fw.release_qubit(mq)

        prev, next = ctx.arms
        log.debug(
            f"{self.node}: SWAP {_OUTCOME_STR[ctx.local_success]}, "
            f"prev={prev.epr.name}@{prev.addr}~{prev.partner.name}, "
            f"next={next.epr.name}@{next.addr}~{next.partner.name}, "
            f"phy_outcome={ctx.phy_outcome}"
        )

        if ctx.local_success:  # swapping succeeded
            self.fw.cnt.n_swapped_s += 1

            # Inform multiplexing scheme.
            self.mux.swapping_succeeded(prev.epr, next.epr, ctx.phy_outcome)

        # Send heralding.
        self._herald_swap_to(0, ctx)
        self._herald_swap_to(1, ctx)

    def _herald_swap_to(self, i: int, ctx: SwapContext):
        """
        Send heralding message.

        Args:
            i: Partner identification, 0 for prev, 1 for next.

        """
        target = ctx.arms[i]
        opposite = ctx.arms[1 - i]

        # Determine expiry time based on heralded information.
        # It's OK to use `.decohere_time` for elementary entanglement as it is local knowledge.
        # TODO Investigate whether `.decohere_time` can be used in swapped entanglement.
        # TODO Incorporate `expiry` from received heralding.
        path_expiry = opposite.epr.decohere_time

        # Send SWAP_UPDATE to the partner.
        self._herald_to(ctx.fib_entry, target, opposite, target.epr.name, path_expiry.time_slot if ctx.local_success else 0)

    def pop_waiting_su(self, qubit: MemoryQubit):
        """
        Invoked by ``Forwarder.qubit_is_entangled()`` after QubitEntangledEvent processing.
        """
        su_args = self.waiting_su.pop(qubit.addr, None)
        if (
            qubit.state != QubitState.RELEASE  # qubit was released due to uninstalled path
            and su_args
        ):
            self.handle_update(*su_args)

    def handle_update(self, msg: SwapUpdateMsg, fib_entry: FibEntry):
        """
        Process an SWAP_UPDATE signaling message.

        Args:
            msg: The SWAP_UPDATE message.
            fib_entry: FIB entry associated with path_id in the message.
        """

        # When using SYNC timing mode but INTERNAL phase is over, qubits are already released.
        if not self.node.timing.is_internal():
            log.debug(f"{self.node}: INT phase is over -> stop swaps")
            return

        old_epr_name = msg["old_epr"]
        new_partner = self.network.get_node(new_partner_name := msg["partner"])
        heralded_success = (heralded_expiry_t := msg["expiry"]) > 0
        self.heralded_expiry[old_epr_name] = heralded_expiry = self.simulator.time(time_slot=heralded_expiry_t)

        old_index, old_rank = fib_entry.find_index_and_swap_rank(old_partner_name := msg["swapping_node"])
        new_index, new_rank = fib_entry.find_index_and_swap_rank(new_partner_name)

        log.debug(
            f"{self.node}: HERALDED {_OUTCOME_STR[heralded_success]}, "
            f"old={old_epr_name}~{old_partner_name}, "
            f"new-partner={new_partner_name}, "
        )

        ctx = HeraldContext(
            fib_entry=fib_entry,
            partner=new_partner,
            partner_index=new_index,
            partner_rank=new_rank,
            partner_attr="src" if new_index < fib_entry.own_idx else "dst",
            old_epr_name=old_epr_name,
            heralded_success=heralded_success,
            heralded_expiry=heralded_expiry,
        )

        if old_rank == fib_entry.own_swap_rank:
            if abs(old_index - ctx.fib_entry.own_idx) != 1:
                raise RuntimeError(f"{self.node} received SwapUpdate from same-ranked non-adjacent node {old_partner_name}")
            self._su_parallel(ctx)
        elif old_rank < fib_entry.own_swap_rank:
            if (epr_pair := self.memory.read(old_epr_name)) is None:
                log.debug(f"{self.node}: EPR {old_epr_name} decohered during SwapUpdate transmission")
                self._cleanup(old_epr_name)
            elif (qubit := epr_pair[0]).state is QubitState.ENTANGLED0:
                # QubitEntangledEvent for the qubit has not been processed.
                # Buffer the SwapUpdate in `self.waiting_su` to be retried in `pop_waiting_su()`.
                #
                # This may happen, for example in S-R-D linear topology, when node R performs a swap
                # as soon as it is receives R-D entanglement, before D is notified.
                # This cannot be resolved with `Event.priority` mechanism because R and D may be notified
                # at different times depending on the link architecture.
                self.waiting_su[qubit.addr] = (msg, fib_entry)
            else:
                self._su_sequential(ctx, qubit)
        else:
            raise RuntimeError(f"{self.node} received SwapUpdate from higher-ranked node {old_partner_name}")

    def _su_parallel(self, ctx: HeraldContext):
        """
        Handle SwapUpdate from a same-ranked node (Parallel Stitching).
        """

        # Retrieve physical outcome deposited by ctx.partner.
        # This would be None if old_epr_name has been used in a local .start() call.
        new_phy_epr = self.remote_swapped.pop(ctx.old_epr_name, None)

        # Retrieve local swap outcome.
        local_phy_epr, local_success = self.local_swapped.pop(ctx.old_epr_name, (None, False))

        if new_phy_epr is None and local_phy_epr is None:
            raise RuntimeError(f"{self.node} received SwapUpdate for {ctx.old_epr_name} but has no recorded")

        # log.debug(f"{self.node}: SwapUpdate-parallel partner={ctx.partner.name} rank={ctx.partner_rank} new_phy={new_phy_epr}")

        if local_phy_epr is None:
            assert new_phy_epr is not None
            # Local swap has not occurred, which suggests that the remote swap was completed sequentially,
            # and the elementary or lower-ranked EPR is not yet available on the opposite arm at the local node.

            if ctx.heralded_success and ctx.heralded_expiry > self.simulator.tc:
                # Remote swapping succeeded, which means we have an EPR with ctx.partner.
                # However, new_phy_epr could be longer than [self,ctx.partner] because ctx.partner may have performed
                # additional swaps that have not been heralded.
                #
                # Given new_phy_epr may contain unheralded information, we cannot save it in memory.
                # Instead, old_epr stays in memory, so that Forwarder.qubit_is_eligible() can find it.
                # We re-deposit new_phy_epr into self.remote_swapped, so that .start() can find it in place of old_epr.
                self.remote_swapped[ctx.old_epr_name] = new_phy_epr
                return
            else:
                # Remote swapping failed or expired.

                # Release the local qubit that has decohered due to failed remote swapping.
                qubit, _ = self.memory.read(ctx.old_epr_name, must=True, remove=True)
                self.fw.release_qubit(qubit, need_remove=False)

                final_phy_epr = new_phy_epr
                overall_success = False

        else:
            # Local swap occurred in parallel with remote swap.
            # Take whatever physical EPR that has a longer chain i.e. is computed later.

            if new_phy_epr and len(cast(list, new_phy_epr.orig_eprs)) > len(cast(list, local_phy_epr.orig_eprs)):
                final_phy_epr = new_phy_epr
            else:
                final_phy_epr = local_phy_epr
            overall_success = ctx.heralded_success and local_success
        del new_phy_epr, local_phy_epr, local_success

        # Identify the adjacent neighbor, in the opposite direction of the herald.
        neigh_offset, neigh_attr = (-1, "src") if ctx.partner_index > ctx.fib_entry.own_idx else (+1, "dst")
        neigh_index = ctx.fib_entry.own_idx + neigh_offset
        neigh_rank = ctx.fib_entry.swap[neigh_index]
        if neigh_rank > ctx.fib_entry.own_swap_rank:
            # We are at the edge of a parallel group and are ready to herald the higher-ranked neighbor.
            neigh_node = self.network.get_node(ctx.fib_entry.route[neigh_index])
            assert final_phy_epr.orig_eprs is not None
            neigh_epr_name = next(e.name for e in final_phy_epr.orig_eprs if getattr(e, neigh_attr) is neigh_node)

            # Deposit physical EPR at the target.
            self._deposit_remote_swapped(neigh_node, neigh_epr_name, final_phy_epr)

            # Herald the higher-ranked neighbor.
            neigh = PartnerInfo(neigh_node, neigh_index, neigh_rank, neigh_attr)
            expiry = 1_000_000_000  # TODO
            self._herald_to(ctx.fib_entry, neigh, ctx, neigh_epr_name, expiry if overall_success else 0)

    def _su_sequential(self, ctx: HeraldContext, qubit: MemoryQubit):
        """
        Handle SwapUpdate from a lower-ranked node (Sequential Extension).
        """

        local_phy_epr, local_success = self.local_swapped.pop(ctx.old_epr_name, (None, False))
        # XXX is this ever non-empty?

        # If heralding message indicates swapping failure or remote expiry, release the qubit.
        if not ctx.heralded_success or ctx.heralded_expiry <= self.simulator.tc:
            self.fw.release_qubit(qubit)
            return

        # Retrieve the physical outcome.
        last_epr_name = ctx.old_epr_name
        while True:
            new_phy_epr = self.remote_swapped.pop(last_epr_name, None)
            if new_phy_epr is None:
                raise RuntimeError(f"physical outcome for {last_epr_name} not found")
            if getattr(new_phy_epr, ctx.partner_attr) is ctx.partner:
                break

        # Save to memory.
        self.memory.write(qubit.addr, new_phy_epr, replace=True)
        log.debug(
            f"{self.node}: SwapUpdate-sequential qubit={qubit.addr} partner={ctx.partner.name} rank={ctx.partner_rank} phy={new_phy_epr}"
        )

        # Start purification, swapping, or consume.
        qubit.purif_rounds = 0
        qubit.state = QubitState.PURIF
        self.fw.qubit_is_purif(qubit, ctx.fib_entry, ctx.partner)

        # if epr_pair := self.memory.read(old_epr_name):
        #     qubit, old_epr = epr_pair
        #     if qubit.state is QubitState.ENTANGLED0:
        #         self.waiting_su[qubit.addr] = (msg, fib_entry)
        #     else:
        #         self._su_sequential(ctx, qubit, old_epr)
        # else:
        #     self._su_parallel()

        # new_epr = self.remote_swapped.pop(old_epr_name) or self.local_swapped[old_epr_name]

        # _, sender_rank = fib_entry.find_index_and_swap_rank(msg["swapping_node"])
        # if fib_entry.own_swap_rank < sender_rank:
        #     log.debug(f"### {self.node}: VERIFY -> rcvd SU from higher-rank node")
        #     return

        # new_epr_name = msg["new_epr"]
        # new_epr = None if new_epr_name is None else self.remote_swapped.pop(new_epr_name)

        # epr_name = msg["epr"]
        # qubit_pair = self.memory.read(epr_name)
        # if qubit_pair is not None:
        #     qubit, _ = qubit_pair
        #     if qubit.state == QubitState.ENTANGLED0:
        #         if new_epr is not None:
        #             self.remote_swapped[cast(str, new_epr_name)] = new_epr
        #         self.waiting_su[qubit.addr] = (msg, fib_entry)
        #         return
        #     self.parallel_swappings.pop(epr_name, None)
        #     self._su_sequential(msg, fib_entry, qubit, new_epr, maybe_purif=(fib_entry.own_swap_rank > sender_rank))
        # elif fib_entry.own_swap_rank == sender_rank and epr_name in self.parallel_swappings:
        #     self._su_parallel(msg, fib_entry, new_epr)
        # else:
        #     log.debug(f"### {self.node}: EPR {epr_name} decohered during SU transmissions")

    # def _su_sequential(
    #     self,
    #     msg: SwapUpdateMsg,
    #     fib_entry: FibEntry,
    #     qubit: MemoryQubit,
    #     new_epr: Entanglement | None,
    #     maybe_purif: bool,
    # ):
    #     """
    #     Process SWAP_UPDATE message where the local MemoryQubit still exists.
    #     This means the swapping was performed sequentially and local MemoryQubit has not decohered.

    #     Args:
    #         maybe_purif: whether the new EPR may enter PURIF state.
    #                      Set to True if own rank is higher than sender rank.
    #     """
    #     if (
    #         new_epr is None  # swapping failed
    #         or new_epr.decohere_time <= self.simulator.tc  # oldest pair decohered
    #     ):
    #         if new_epr:
    #             log.debug(f"{self.node}: NEW EPR {new_epr} decohered during SU transmissions")
    #         # Inform LinkLayer that the memory qubit has been released.
    #         self.fw.release_qubit(qubit, need_remove=True)
    #         return

    #     # Update old EPR with new EPR (fidelity and partner).
    #     self.memory.write(qubit.addr, new_epr, replace=True)

    #     if maybe_purif:
    #         # If own rank is higher than sender rank but lower than new partner rank,
    #         # it is our turn to purify the qubit and progress toward swapping.
    #         qubit.purif_rounds = 0
    #         qubit.state = QubitState.PURIF
    #         partner = self.network.get_node(msg["partner"])
    #         self.fw.qubit_is_purif(qubit, fib_entry, partner)

    # def _su_parallel(self, msg: SwapUpdateMsg, fib_entry: FibEntry, new_epr: Entanglement | None):
    #     """
    #     Process SWAP_UPDATE message during parallel swapping.
    #     """
    #     shared_epr, other_epr, my_new_epr = self.parallel_swappings.pop(msg["epr"])
    #     _ = shared_epr

    #     # safety in statistical mux to avoid conflictual swappings on different paths
    #     if self.mux.su_parallel_has_conflict(my_new_epr, msg["path_id"]):
    #         self.fw.cnt.n_swap_conflict += 1
    #         return

    #     # msg["swapping_node"] is the node that performed swapping and sent this message.
    #     # Assuming swapping_node is to the right of own node, various nodes and EPRs are as follows:
    #     #
    #     # destination-------own--------swapping_node----partner
    #     #      |             |~~shared_epr~~|            |
    #     #      |~~other_epr~~|              |            |
    #     #      |~~~~~~~~~~my_new_epr~~~~~~~~|            |
    #     #      |             |~~~~~~~~~~new_epr~~~~~~~~~~|
    #     #      |~~~~~~~~~~~~~~~merged_epr~~~~~~~~~~~~~~~~|

    #     if (
    #         new_epr is None  # swapping failed
    #         or new_epr.decohere_time <= self.simulator.tc  # oldest pair decohered
    #     ):
    #         # Determine the "destination".
    #         if other_epr.dst == self.node:  # destination is to the left of own node
    #             destination = other_epr.src
    #         else:  # destination is to the right of own node
    #             destination = other_epr.dst
    #         assert destination is not None

    #         # Inform the "destination" that swapping has failed.
    #         su_msg: SwapUpdateMsg = {
    #             "cmd": "SWAP_UPDATE",
    #             "path_id": msg["path_id"],
    #             "swapping_node": msg["swapping_node"],
    #             "partner": msg["partner"],
    #             "epr": my_new_epr.name,
    #             "new_epr": None,
    #         }
    #         self.fw.send_msg(destination, su_msg, fib_entry)
    #         return

    #     # The swapping_node successfully swapped in parallel with this node.
    #     # Determine the "destination" and "partner".
    #     # Merge the two swaps (physically already happened).
    #     new_epr.read = True
    #     if other_epr.dst == self.node:  # destination is to the left of own node
    #         merged_epr, _ = Entanglement.swap(other_epr, new_epr, now=self.simulator.tc)
    #         partner = cast(QNode, new_epr.dst)
    #         destination = cast(QNode, other_epr.src)
    #     else:  # destination is to the right of own node
    #         merged_epr, _ = Entanglement.swap(new_epr, other_epr, now=self.simulator.tc)
    #         partner = cast(QNode, new_epr.src)
    #         destination = cast(QNode, other_epr.dst)
    #     assert partner.name == msg["partner"]

    #     if not merged_epr.is_decohered:  # XXX is_decohered is hidden variable and cannot be accessed
    #         self.fw.cnt.n_swapped_p += 1

    #         # adjust EPR paths for dynamic EPR affectation and statistical mux
    #         self.mux.su_parallel_succeeded(merged_epr, new_epr, other_epr)

    #         # Inform the "destination" of the swap result and new "partner".
    #         self._deposit_remote_swapped(destination, merged_epr)

    #     su_msg: SwapUpdateMsg = {
    #         "cmd": "SWAP_UPDATE",
    #         "path_id": msg["path_id"],
    #         "swapping_node": msg["swapping_node"],
    #         "partner": partner.name,
    #         "epr": my_new_epr.name,
    #         "new_epr": None if merged_epr is None else merged_epr.name,
    #     }
    #     self.fw.send_msg(destination, su_msg, fib_entry)

    #     # Update records to support potential parallel swapping with "partner".
    #     _, p_rank = fib_entry.find_index_and_swap_rank(partner.name)
    #     if fib_entry.own_swap_rank == p_rank and merged_epr is not None:
    #         self.parallel_swappings[new_epr.name] = (new_epr, other_epr, merged_epr)
