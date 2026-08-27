from collections.abc import Iterable

from mqns.entity.memory import PathDirection
from mqns.entity.qchannel import QuantumChannel
from mqns.network.fw.fib import FibPath, FibRequest
from mqns.network.fw.fw_module import ForwarderModule
from mqns.network.fw.message import PathReachEprCountMsg


class ForwarderNorthbound(ForwarderModule):
    """
    Northbound interface of the forwarder to communicate with the controller.
    """

    def iter_adjacency(self, fp: FibPath) -> Iterable[tuple[PathDirection, QuantumChannel]]:
        """Iterate over quantum channels connected to adjacent nodes."""
        for dir, route_offset in (PathDirection.L, -1), (PathDirection.R, +1):
            neigh_idx = fp.own_idx + route_offset
            if neigh_idx in (-1, len(fp.route)):
                continue
            neigh = self.network.get_node(fp.route[neigh_idx])
            ch = self.node.get_qchannel(neigh)
            yield dir, ch

    def send_reach_epr_count(self, fr: FibRequest) -> None:
        """Send PATH_REACH_EPR_COUNT message to the controller."""
        msg = PathReachEprCountMsg(
            cmd="PATH_REACH_EPR_COUNT",
            req_id=fr.req_id,
        )
        self.send_ctrl(msg)
