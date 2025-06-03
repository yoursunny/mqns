from qns.network.network import QuantumNetwork
from qns.network.route import DijkstraRouteAlgorithm
from qns.network.topology import LinearTopology


def test_dijkstra():
    net = QuantumNetwork(topo=LinearTopology(4), route=DijkstraRouteAlgorithm())
    net.build_route()

    n1 = net.get_node("n1")
    n2 = net.get_node("n2")
    n3 = net.get_node("n3")
    n4 = net.get_node("n4")

    r11 = net.query_route(n1, n1)
    assert len(r11) == 0

    r12 = net.query_route(n1, n2)
    assert len(r12) == 1
    assert r12[0] == (1, n2, [n1, n2])

    r14 = net.query_route(n1, n4)
    assert len(r14) == 1
    assert r14[0] == (3, n2, [n1, n2, n3, n4])
