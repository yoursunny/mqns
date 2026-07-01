from typing import cast

type NodePair = str | tuple[str, str]
"""
Two node names on a channel or routing path.
This could be either a tuple of two node names, or a string delimited by hyphen (``-``).
"""


def split_node_pair(np: NodePair) -> tuple[str, str]:
    if isinstance(np, str):
        tokens = np.split("-")
        if len(tokens) != 2:
            raise ValueError(f"expect two node names in '{np}'")
        return cast(tuple[str, str], tuple(tokens))
    return np
