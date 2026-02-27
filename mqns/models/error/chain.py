from collections.abc import Iterable
from typing import Self, override

from mqns.models.error.error import ErrorModel


class ChainErrorModel(ErrorModel):
    """
    Chain that composites multiple error models.

    The ``set`` method sets the same parameters onto each enclosed error model.
    To set different parameters, call ``set`` on each enclosed error model directly.
    """

    def __init__(self, errors: Iterable[ErrorModel]):
        """
        Constructor:

        Args:
            errors: list of error models applied in sequential order.
        """
        self.errors = list(errors)
        super().__init__("CHAIN(" + ",".join(m.name for m in self.errors) + ")")

    @override
    def _set(self, **kwargs) -> Self:
        for m in self.errors:
            m.set(**kwargs)
        return self

    @override
    def qubit(self, q) -> None:
        for m in self.errors:
            m.qubit(q)

    @override
    def werner(self, q) -> None:
        for m in self.errors:
            m.werner(q)

    @override
    def mixed(self, q) -> None:
        for m in self.errors:
            m.mixed(q)
