import os
import sys
from typing import Any, Literal, cast

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

_next_seed_env: list[int] = []
"""
This list must contain zero or one integer.
It is populated by the first ``rng.reseed("env")`` call, and incremented on subsequent calls.
"""


def _reseed_env() -> int | None:
    if _next_seed_env:
        seed = _next_seed_env[0]
    elif env := os.getenv("MQNS_SEED"):
        seed = int(env)
        _next_seed_env.append(seed)
    else:
        return None

    if "pytest" in sys.modules:
        print(f"MQNS_SEED={seed}")

    _next_seed_env[0] = seed + 1
    return seed


class RngUtils:
    def reseed(self, seed: int | None | Literal["env"]):
        """
        Reseed the random number generator.

        The new ``seed`` could be one of:

        * Integer seed.
        * ``None`` for NumPy default.
        * "env": set the first seed with MQNS_SEED environment variable,
          auto-increment subsequent seeds.
          If MQNS_SEED is unset, this is equivalent to ``None``.

        When "env" mode is used with pytest, the actual seed value is printed to stdout.
        Combined with ``pytest-repeat``, this enables detecting seed-specific test failures.
        To run a test case repeatedly with increasing seeds:

            MQNS_SEED=0 pytest test_file.py::test_case --count=100 -x

        To run a test case with a specific seed:

            MQNS_SEED=47 pytest test_file.py::test_case
        """
        global _rng
        if seed == "env":
            seed = _reseed_env()

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
