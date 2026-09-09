import itertools
from collections import defaultdict, deque

from mqns.network.reactive.message import LinkStateEntry


class TopoLinkState:
    """
    Topology link state -- controller's view of available EPRs in the network.
    """

    def __init__(self):
        self.d = defaultdict[tuple[str, str], deque[str]](deque)
        """
        Key: node names, sorted.
        Value: entanglement reservation keys.
        """

    def clear(self) -> None:
        self.d.clear()

    def add(self, entry: LinkStateEntry) -> None:
        """
        Save a link state entry.
        """
        a = entry["node"]
        b = entry["neighbor"]
        if a < b:
            self.d[a, b].append(entry["qubit"])

    def try_consume(self, path: list[str]) -> list[str] | None:
        """
        Attempt to match a computed route with available entanglements.
        The entanglements are removed this table only if every link along the path has an entanglement.
        """
        link_etgs: list[deque[str]] = []

        for a, b in itertools.pairwise(path):
            etgs = self.d.get((a, b) if a < b else (b, a))
            if not etgs:
                return None
            link_etgs.append(etgs)

        return [etgs.popleft() for etgs in link_etgs]
