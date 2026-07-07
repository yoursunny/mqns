from collections.abc import Sequence

from mqns.entity.node import NodePair
from mqns.network.network.network import QuantumNetwork
from mqns.network.network.request import Request
from mqns.utils import rng


def _violates_endpoint_internal(path0: Sequence[str], path1: Sequence[str]) -> bool:
    """
    Determine whether any node is treated as endpoint in one path and internal in another path.
    """
    for a, b in (path0, path1), (path1, path0):
        if a[0] in b[1:-1] or a[-1] in b[1:-1]:
            return True
    return False


def generate_random_requests(
    net: QuantumNetwork,
    n: int,
    *,
    allow_overlay=False,
    allow_endpoint_internal=False,
    metric_min: float = 1,
    metric_max: float = 10,
) -> list[Request]:
    """
    Generate random requests in a network.

    Args:
        net: The quantum network.
        n: Number of requests.
        allow_overlay: If True, the same node may appear as endpoints in multiple requests.
        allow_endpoint_internal: If True, the same node may appear as an endpoint in one path
                                 and an internal node in another path.
        metric_min: Minimum routing metric, often number of hops (inclusive).
        metric_max: Maximum routing metric, often number of hops (inclusive).
    """
    nnodes = len(net.nodes)
    if not allow_overlay and n * 2 > nnodes:
        raise ValueError("Too many requests")
    net.build_route()

    requests: list[Request] = []
    used_endpoints = set[int]()
    used_paths: list[list[str]] = []

    while (req_id := len(requests)) < n:
        src = rng.integers(0, nnodes, dtype=int)
        dst = rng.integers(0, nnodes, dtype=int)
        if src == dst:
            continue
        if not allow_overlay and (src in used_endpoints or dst in used_endpoints):
            continue

        np: NodePair = net.nodes[src].name, net.nodes[dst].name
        routes = net.query_route(*np, error_on_empty=False)
        if not routes:
            continue

        route = routes[0]
        if not (metric_min <= route.metric <= metric_max):
            continue

        if not allow_endpoint_internal and any(_violates_endpoint_internal(path0, route.path) for path0 in used_paths):
            continue

        requests.append(Request(np).path(req_id=req_id))
        used_endpoints.add(src)
        used_endpoints.add(dst)
        used_paths.append(route.path)

    return requests
