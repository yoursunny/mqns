from collections.abc import Callable
from typing import Any, cast, override

import numpy.random as npr

_rng = npr.default_rng()
"""
Real rng instance.
This may be re-assigned.
"""

class RngUtils:
    def reseed(self, seed: int|None):
        """
        Reseed the random number generator.
        """
        global _rng
        _rng = npr.default_rng(npr.PCG64(seed))



class RngProxy(RngUtils):
    """
    Proxy class for global rng.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_rng, name)


class RngPublic(npr.Generator, RngUtils):
    def __init__(self):
        assert False


rng = cast(RngPublic, RngProxy())
"""
Global random number generator.
"""


def set_seed(seed: int | None):
    """
    Reseed the random number generator.
    """
    global _rng
    _rng = npr.default_rng(npr.PCG64(seed))


class FixedRng(npr.Generator):
    """
    Random number generator that returns fixed values.

    This is primarily useful for unit testing.
    """

    def __init__(self, v: Callable[[], float] | float | None = None):
        super().__init__(rng.bit_generator)
        self._v = (lambda: v) if isinstance(v, (int, float)) else v

    @override
    def random(self, *args, **kwargs) -> Any:
        return self._v() if self._v else super().random(*args, **kwargs)

    @override
    def uniform(self, *args, **kwargs) -> Any:
        return self._v() if self._v else super().uniform(*args, **kwargs)
