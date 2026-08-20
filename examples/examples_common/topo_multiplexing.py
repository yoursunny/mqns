from mqns.network.builder import NetworkBuilder

END_NODES = ("A", "B", "C", "D", "G", "H", "I", "K", "L", "M")
"""Nodes with degree=1."""

TX_QUBITS = 50
RX_QUBITS = 32


def define_topo(b: NetworkBuilder) -> None:
    """
    Build a 13-node tree/star topology characterized by two highly contested central backbone routing links.

    .. figure:: /_static/examples/multiplexing.svg
        :alt: topology diagram
        :align: center
        :width: 100%
    """
    b.topo(
        channels=[
            # left spokes -> E
            (("A", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("B", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("C", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            (("D", "E"), 30, (TX_QUBITS, RX_QUBITS)),
            # middle trunks
            (("E", "F"), 30, (RX_QUBITS, TX_QUBITS)),
            (("F", "J"), 30, (TX_QUBITS, RX_QUBITS)),
            # right spokes from J
            (("J", "K"), 30, (TX_QUBITS, RX_QUBITS)),
            (("J", "L"), 30, (TX_QUBITS, RX_QUBITS)),
            (("J", "M"), 30, (TX_QUBITS, RX_QUBITS)),
            # bottom spokes from F
            (("G", "F"), 30, (RX_QUBITS, TX_QUBITS)),
            (("F", "H"), 30, (TX_QUBITS, RX_QUBITS)),
            (("F", "I"), 30, (TX_QUBITS, RX_QUBITS)),
        ],
        fiber_alpha=0.17,
        eta_d=0.5,
        eta_s=0.8,
        t_cohere=0.1,
    )
