from mqns.network.fw.fib import FibRequest
from mqns.network.fw.fw_module import ForwarderModule
from mqns.network.fw.message import PathReachEprCountMsg


class ForwarderNorthbound(ForwarderModule):
    """
    Northbound interface of the forwarder to communicate with the controller.
    """

    def send_reach_epr_count(self, fr: FibRequest) -> None:
        """Send PATH_REACH_EPR_COUNT message to the controller."""
        msg = PathReachEprCountMsg(
            cmd="PATH_REACH_EPR_COUNT",
            req_id=fr.req_id,
        )
        self.send_ctrl(msg)
