from typing import Any, cast

import numpy.random as npr

_rng = npr.default_rng()
"""
Real rng instance.
This may be re-assigned.
"""

_FAST_METHODS = (
    "choice",
    "exponential",
    "geometric",
    "integers",
    "normal",
    "random",
    "shuffle",
    "uniform",
)
"""
Method names on ``rng`` to avoid getattr overhead.
"""


class RngUtils:
    def reseed(self, seed: int | None):
        """
        Reseed the random number generator.
        """
        global _rng
        _rng = npr.default_rng(npr.PCG64(seed))

        if isinstance(self, RngProxy):
            self._update_fast_methods()


class RngProxy(RngUtils):
    """
    Proxy class for global rng.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_rng, name)

    def _update_fast_methods(self) -> None:
        for method in _FAST_METHODS:
            self.__dict__[method] = getattr(_rng, method)


class RngPublic(npr.Generator, RngUtils):
    """
    Global random number generator, public API declaration.
    """


rng = cast(RngPublic, RngProxy())
"""
Global random number generator.
"""

cast(RngProxy, rng)._update_fast_methods()
