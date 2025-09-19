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

import math
from typing import Any, cast

import numpy as np
from scipy.sparse.csgraph import dijkstra

from qns.network.route.route import ChannelT, MetricFunc, NetworkRouteError, NodeT, RouteImpl, make_csr


class DijkstraRouteAlgorithm(RouteImpl[NodeT, ChannelT]):
    """This is the Dijkstra algorithm implementation"""

    INF = math.inf

    def __init__(self, name: str = "dijkstra", metric_func: MetricFunc | None = None) -> None:
        """
        Args:
            name: Name of the routing algorithm (default: "dijkstra").
            metric_func: Function returning the metric (weight) for each channel.
                Defaults to a constant function m(l) = 1.
        """
        self.name = name
        self.route_table: dict[NodeT, dict[NodeT, tuple[float, list[NodeT]]]] = {}

        if metric_func is None:
            self.metric_func = lambda _: 1  # hop count
            self.unweighted = True
        else:
            self.metric_func = metric_func
            self.unweighted = False

    def build(self, nodes: list[NodeT], channels: list[ChannelT]):
        """
        Build the routing table using SciPy's csgraph Dijkstra on a CSR adjacency.

        Args:
            nodes: a list of quantum nodes or classic nodes
            channels: a list of quantum channels or classic channels
        """

        # build adjacency matrix
        csr_adj = make_csr(nodes, channels, self.metric_func)

        # unweighted=True -> hop count; directed=False for undirected topologies
        dist, preds = dijkstra(
            csr_adj,
            directed=False,
            unweighted=self.unweighted,
            return_predecessors=True,
        )

        # Reconstruct path helper
        def _reconstruct_path(src_idx: int, dst_idx: int) -> list[NodeT]:
            # Backtrack from dst to src using predecessors
            path_idx: list[int] = []
            u = dst_idx
            while u not in (-9999, src_idx):
                path_idx.append(u)
                u = preds[src_idx, u]
            path_idx.append(src_idx)
            return [nodes[i] for i in path_idx]

        self.route_table.clear()

        # For each source node, create the per-destination entry
        for src_idx, src_node in enumerate(nodes):
            dest_entry: dict[NodeT, Any] = {}

            for dst_idx, dst_node in enumerate(nodes):
                if src_idx == dst_idx:
                    # Source to itself
                    dest_entry[dst_node] = [0.0, [dst_node]]
                    continue

                hop = dist[src_idx, dst_idx]
                if np.isinf(hop):  # Unreachable
                    dest_entry[dst_node] = [self.INF, [dst_node]]
                else:
                    path_nodes = _reconstruct_path(src_idx, dst_idx)
                    dest_entry[dst_node] = [hop, path_nodes]

            self.route_table[src_node] = dest_entry

    def build2(self, nodes: list[NodeT], channels: list[ChannelT]):
        """
        Build the routing table using a local implementation of Dijkstra algorithm.

        Args:
            nodes: a list of quantum nodes or classic nodes
            channels: a list of quantum channels or classic channels
        """

        self.route_table.clear()

        for source_node in nodes:
            visited_nodes: list[NodeT] = []
            unvisited_nodes: list[NodeT] = [node for node in nodes]

            # tentative distance and path table for this source
            tentative: dict[NodeT, Any] = {}
            for destination_node in nodes:
                if destination_node == source_node:
                    tentative[source_node] = [0.0, []]
                else:
                    tentative[destination_node] = [self.INF, [destination_node]]

            # Dijkstra-like loop
            while unvisited_nodes:
                # pick one node to init
                current_node = unvisited_nodes[0]
                current_distance = tentative[current_node][0]

                for candidate in unvisited_nodes:
                    if tentative[candidate][0] < current_distance:
                        current_node = candidate
                        current_distance = tentative[candidate][0]

                visited_nodes.append(current_node)
                unvisited_nodes.remove(current_node)

                # relax edges for neighbors
                for channel in channels:
                    if current_node not in channel.node_list:
                        continue
                    if len(channel.node_list) < 2:
                        raise NetworkRouteError("broken link")

                    idx_in_channel = cast(list[NodeT], channel.node_list).index(current_node)
                    neighbor_node = cast(list[NodeT], channel.node_list)[1 - idx_in_channel]

                    if neighbor_node in unvisited_nodes and tentative[neighbor_node][0] > tentative[current_node][
                        0
                    ] + self.metric_func(channel):
                        tentative[neighbor_node] = [
                            tentative[current_node][0] + self.metric_func(channel),
                            [current_node] + tentative[current_node][1],
                        ]

            # finalize paths
            for destination_node in nodes:
                tentative[destination_node][1] = [destination_node] + tentative[destination_node][1]

            self.route_table[source_node] = tentative

    def query(self, src: NodeT, dest: NodeT) -> list[tuple[float, NodeT, list[NodeT]]]:
        ls = self.route_table.get(src, None)
        if ls is None:
            return []
        le = ls.get(dest, None)
        if le is None:
            return []
        try:
            metric, path = le
            path = path.copy()
            path.reverse()
            if len(path) <= 1 or metric == self.INF:  # unreachable
                next_hop = None
                return []
            else:
                next_hop = path[1]
                return [(metric, next_hop, path)]
        except Exception:
            return []
