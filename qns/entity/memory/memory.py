#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
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

from typing import List, Optional, Union, Tuple
from qns.models.delay.constdelay import ConstantDelayModel
from qns.models.delay.delay import DelayModel
from qns.simulator.simulator import Simulator
from qns.simulator.ts import Time
from qns.models.core.backend import QuantumModel
from qns.entity.entity import Entity
from qns.entity.node.qnode import QNode
from qns.entity.memory.memory_qubit import MemoryQubit, QubitState
from qns.simulator.event import Event, func_to_event
import qns.utils.log as log


class OutOfMemoryException(Exception):
    """
    The exception that the memory is full
    """
    pass


class QuantumMemory(Entity):
    """
    Quantum memory stores qubits or entangled pairs.

    It has two modes:
        Synchronous mode, users can use the ``read`` and ``write`` function to operate the memory directly without delay
        Asynchronous mode, users can use events to operate memories asynchronously
    """
    def __init__(self, name: str = None, node: QNode = None,
                 capacity: int = 0, decoherence_rate: Optional[float] = 0,
                 store_error_model_args: dict = {}, delay: Union[float, DelayModel] = 0):
        """
        Args:
            name (str): its name
            node (QNode): the quantum node that equips this memory
            capacity (int): the capacity of this quantum memory. 0 represents unlimited.
            delay (Union[float,DelayModel]): the read and write delay in second, or a ``DelayModel``
            decoherence_rate (float): the decoherence rate of this memory that will pass to the store_error_model
            store_error_model_args (dict): the parameters that will pass to the store_error_model
        """
        super().__init__(name=name)
        self.node = node
        self.capacity = capacity
        self.delay_model = delay if isinstance(delay, DelayModel) else ConstantDelayModel(delay=delay)

        if self.capacity > 0:
            self._storage: List[Tuple[MemoryQubit, Optional[QuantumModel]]] = [
                    (MemoryQubit(addr), None) for addr in range(self.capacity)
            ]
            self._store_time: List[Optional[Time]] = [None] * self.capacity
        else:      # should not use this case
            print("Error: unlimited memory capacity not supported")
            return

        self._usage = 0
        self.decoherence_rate = decoherence_rate
        self.store_error_model_args = store_error_model_args
        
        self.link_layer = None
        
        self.pending_decohere_events = {}


    def install(self, simulator: Simulator) -> None:
        from qns.network.protocol.link_layer import LinkLayer
        super().install(simulator)
        ll_apps = self.node.get_apps(LinkLayer)
        if ll_apps:
            self.link_layer = ll_apps[0]
        else:
            raise Exception("No LinkLayer protocol found")

    def _search(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> int:
        index = -1
        if address is not None:
            for idx, (qubit, _) in enumerate(self._storage):
                if qubit.addr == address:
                    return idx
        elif isinstance(key, int):
            if self.capacity == 0 and key >= 0 and key < self._usage:
                index = key
            elif key >= 0 and key < self.capacity and self._storage[key][1] is not None:
                index = key
        elif isinstance(key, QuantumModel):
            for idx, (_, data) in enumerate(self._storage):
                if data is None:
                    continue
                if data == key:
                    return idx
        elif isinstance(key, str):
            for idx, (_, data) in enumerate(self._storage):
                if data is None:
                    continue
                if data.name == key:
                    return idx
        return index

    def get(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Tuple[MemoryQubit, Optional[QuantumModel]]:
        """
        get a qubit from the memory but without removing it from the memory

        Args:
            key (Union[QuantumModel, str, int]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        idx = self._search(key=key, address=address)
        if idx != -1:
            return self._storage[idx]
        else:
            return None

    def get_store_time(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Optional[Time]:
        """
        get the store time of a qubit from the memory

        Args:
            key (Union[QuantumModel, str, int]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        try:
            idx = self._search(key, address)
            if idx != -1:
                return self._store_time[idx]
            else:
                return None
        except IndexError:
            return None

    def read(self, key: Optional[Union[QuantumModel, str, int]] = None, address: Optional[int] = None) -> Tuple[MemoryQubit, Optional[QuantumModel]]:
        """
        Destructive reading of a qubit from the memory

        Args:
            key (Union[QuantumModel, str]): the key. It can be a QuantumModel object,
                its name or the index number.
        """
        idx = self._search(key=key, address=address)
        if idx == -1:
            return None

        (qubit, data) = self._storage[idx]
        store_time = self._store_time[idx]
        self._usage -= 1

        self._storage[idx] = (self._storage[idx][0], None)
        self._store_time[idx] = None

        t_now = self._simulator.current_time
        sec_diff = t_now.sec - store_time.sec
        data.store_error_model(t=sec_diff, decoherence_rate=self.decoherence_rate, **self.store_error_model_args)

        # cancel scheduled decoherence event
        event = self.pending_decohere_events[data.name]
        event.cancel()
        self.pending_decohere_events.pop(data.name)

        return (qubit, data)

    def write(self, qm: QuantumModel, pid: Optional[int] = None, address: Optional[int] = None, key: str = None, delay: float = 0) -> Optional[MemoryQubit]:
        """
        The API for storing a qubit to the memory

        Args:
            qm (QuantumModel): the `QuantumModel`, could be a qubit or an entangled pair

        Returns:
            bool: whether the qubit is stored successfully
        """
        if self.is_full():
            return None

        idx = -1
        for i, (q, v) in enumerate(self._storage):
            if v is None and (key is None or key == q.active):           # Check if the slot is empty
                if (pid is None or q.pid == pid) and (address is None or q.addr == address):
                    idx = i
                    break
        if idx == -1:
            return None

        self._storage[idx] = (self._storage[idx][0], qm)
        self._store_time[idx] = self._simulator.current_time - Time(sec=delay)
        self._usage += 1

        # schedule an event at T_coh to decohere the qubit
        t = self._simulator.tc - Time(sec=delay) + Time(sec = 1 / self.decoherence_rate)   # align store time with EPR generation time at sender
        event = func_to_event(t, self.decohere_qubit, by=self, qubit=self._storage[idx][0], qm=qm)
        self.pending_decohere_events[qm.name] = event
        self._simulator.add_event(event)

        qm.decoherence_time = t
        return self._storage[idx][0]    # return the memory qubit

    def update(self, old_qm: str, new_qm: QuantumModel) -> bool:
        """
        The API for updating a qubit with a new data without resetting coherence time

        Args:
            old_qm (QuantumModel): the `QuantumModel` to update, could be a qubit or an entangled pair
            new_qm (QuantumModel): the `QuantumModel` to store, could be a qubit or an entangled pair

        Returns:
            bool: whether the qubit is updated successfully. Returns False if old_qm does not exist anymore.
        """
        idx = self._search(key=old_qm)
        if idx == -1:
            if old_qm in self.pending_decohere_events:
                print(f"UNEXPECTED ==> decohere event not cleared")
                old_event = self.pending_decohere_events[old_qm]
                old_event.cancel()
                self.pending_decohere_events.pop(old_qm)
            return False

        self._storage[idx] = (self._storage[idx][0], new_qm)

        old_event = self.pending_decohere_events[old_qm]
        old_event.cancel()
        self.pending_decohere_events.pop(old_qm)

        # schedule an event at old T_coh to decohere the qubit
        new_event = func_to_event(new_qm.decoherence_time, self.decohere_qubit, by=self, qubit=self._storage[idx][0], qm=new_qm)
        self.pending_decohere_events[new_qm.name] = new_event
        self._simulator.add_event(new_event)
        return True

    def clear(self) -> None:
        """
        Clear all qubits in the memory
        """
        for idx, (qubit, _) in enumerate(self._storage):
            qubit.fsm.to_release()
            self._storage[idx] = (qubit, None)
            self._store_time[idx] = None
        self._usage = 0
        for _, event in self.pending_decohere_events.items():
            event.cancel()
        self.pending_decohere_events = {}

    def allocate(self, path_id: int) -> int:
        """ 
        Allocate a qubit to a path
        """
        for (qubit,_) in self._storage:
            if qubit.pid is None:
                qubit.allocate(path_id) 
                return qubit.addr
        return -1

    def deallocate(self, address: int) -> bool:
        """ 
        De-allocate a qubit from any path
        """
        for (qubit,_) in self._storage:
            if qubit.addr == address:
                qubit.deallocate()    
                return True
        return False

    def search_eligible_qubits(self, qchannel: str, pid: int = None) -> List[Tuple[MemoryQubit, QuantumModel]]:
        qubits = []
        for (qubit, data) in self._storage:
            if data and qubit.fsm.state == QubitState.ELIGIBLE and qubit.pid == pid and qubit.qchannel.name != qchannel:
                qubits.append((qubit, data))
        return qubits

    def search_path_qubits(self, pid: int = None) -> List[MemoryQubit]:
        qubits = []
        for (qubit, data) in self._storage:
            if not data and qubit.pid == pid and not qubit.active:
                qubits.append(qubit)
        return qubits
    
    def search_purif_qubits(self, curr_qubit_addr: int, partner: str, qchannel: str, pid: int = None, purif_rounds: int = 0) -> List[Tuple[MemoryQubit, QuantumModel]]:
        # recurrence purif -> pairs must be of equal rounds
        # log.debug(f"partner={partner}, qchannel={qchannel}, pid={pid}, purif_rounds={purif_rounds}")
        qubits = []
        for (qubit, data) in self._storage:
            #if data:
                # log.debug(f"----- {qubit} | {data.src.name}-{data.dst.name}")
            if qubit.addr != curr_qubit_addr and data and qubit.fsm.state == QubitState.PURIF and qubit.pid == pid \
                and qubit.qchannel.name == qchannel and (data.src.name == partner or data.dst.name == partner) \
                and qubit.purif_rounds == purif_rounds:
                qubits.append((qubit, data))
        return qubits
    
    # for dynamic capacity allocation
    def assign(self, ch) -> int:
        """ 
        Assign a qubit to a particular quantum channel
        """
        for (qubit,_) in self._storage:
            if qubit.qchannel is None:
                qubit.assign(ch) 
                return qubit.addr
        return -1

    def unassign(self, address: int) -> bool:
        """ 
        Unassign a qubit from any quantum channel
        """
        for (qubit,_) in self._storage:
            if qubit.addr == address:
                qubit.unassign()
                return True
        return False
    
    def get_channel_qubits(self, ch_name: str) -> List[MemoryQubit]:
        qubits = []
        for (qubit, data) in self._storage:
            if qubit.qchannel and qubit.qchannel.name == ch_name:
                qubits.append((qubit, data))
        return qubits

    def is_full(self) -> bool:
        """
        check whether the memory is full
        """
        return self.capacity > 0 and self._usage >= self.capacity

    @property
    def count(self) -> int:
        """
        return the current memory usage
        """
        return self._usage
    
    @property
    def free(self) -> int:
        """
        return the number of non-allocated memory qubits
        """
        free = self.capacity
        for (qubit,_) in self._storage:
            if qubit.pid:
                free-=1
        return free

    def handle(self, event: Event) -> None:
        from qns.entity.memory.event import MemoryReadRequestEvent, MemoryReadResponseEvent, \
                                            MemoryWriteRequestEvent, MemoryWriteResponseEvent
        if isinstance(event, MemoryReadRequestEvent):
            key = event.key
            # operate qubits and get measure results
            result = self.read(key)

            t = self._simulator.tc + self._simulator.time(sec=self.delay_model.calculate())
            response = MemoryReadResponseEvent(node=self.node, result=result, request=event, t=t, by=self)
            self._simulator.add_event(response)
        elif isinstance(event, MemoryWriteRequestEvent):
            qubit = event.qubit
            result = self.write(qubit)
            t = self._simulator.tc + self._simulator.time(sec=self.delay_model.calculate())
            response = MemoryWriteResponseEvent(node=self.node, result=result, request=event, t=t, by=self)
            self._simulator.add_event(response)


    def decohere_qubit(self, qubit: MemoryQubit, qm: QuantumModel):
        # we try to read EPR (not qubit addr) to make sure we are dealing with this particular EPR:
        # - if qubit has been in swap/purify -> L3 should have released qubit and notified L2.
        # - if qubit has been re-entangled, self.read for EPR.name will not find it, so no notification.
        qm.is_decoherenced = True
        if self.read(key=qm):
            # self.pending_decohere_events.pop(qm.name)    # already done by read()
            log.debug(f"{self.node}: EPR decohered -> {qm.name} {qm.src}-{qm.dst}")
            qubit.fsm.to_release()
            from qns.network.protocol.event import QubitDecoheredEvent
            t = self._simulator.tc # + self._simulator.time(sec=0)
            event = QubitDecoheredEvent(link_layer=self.link_layer, qubit=qubit, t=t, by=self)
            self._simulator.add_event(event)


    def __repr__(self) -> str:
        if self.name is not None:
            return "<memory "+self.name+">"
        return super().__repr__()
