from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast


def unwrap[T](var: T | None) -> T:
    """
    Assert a variable is not None.
    """
    assert var is not None
    return var


def unwrap_cast[T](var: T | None) -> T:
    """
    Cast a variable to non-None null.
    """
    return cast(T, var)


class DecoratorDispatchBuilder[K, F: Callable]:
    """
    Helper for building a decorator-based dispatcher.
    """

    def __init__(
        self,
        *,
        classvar_name: str,
        decorator_attr: str,
    ):
        self.classvar_name = classvar_name
        self.decorator_attr = decorator_attr

    def make_decorator(self, key: K) -> Callable[[F], F]:
        """
        Make a decorator that decorates a class method to be dispatched on a specific key.

        Args:
            key: Dispatching key.
        """

        def decorator(target: F) -> F:
            setattr(target, self.decorator_attr, key)
            return target

        return decorator

    def gather(self, cls: type, nested_base: type | None = None) -> dict[K, list[F]]:
        """
        Gather decorated methods in the class hierarchy.

        Args:
            cls: Concrete class.
            nested_base: If specified, also gather decorated methods from member variables
                of ``cls`` class if the member has ``nested_base`` as a base class.
                The member variable must be declared at class level with a type hint referencing
                its type; forward references and annotations are not supported.
        """
        d = defaultdict[K, list[F]](list)

        if nested_base is not None:
            self._gather_nested(d, cls, nested_base)

        for base in cls.mro()[:-1]:  # skipping last item `object`
            self._gather_base(d, base)

        return dict(d)

    def _gather_nested(self, d: defaultdict[K, list[F]], cls: type, nested_base: type) -> None:
        for attr_name, attr_type in cls.__annotations__.items():
            if not (isinstance(attr_type, type) and issubclass(attr_type, nested_base)):
                continue
            for k, l in cast(dict[K, list[F]], getattr(attr_type, self.classvar_name)).items():
                d[k] += (self._make_wrapper(attr_name, func) for func in l)

    def _gather_base(self, d: defaultdict[K, list[F]], base: type) -> None:
        if b := cast(dict[K, list[F]], base.__dict__.get(self.classvar_name)):
            for k, l in b.items():
                d[k] += l
            return

        for func in base.__dict__.values():
            if (key := getattr(func, self.decorator_attr, None)) is not None:
                d[key].append(func)

    @staticmethod
    def _make_wrapper(attr_name: str, f: Callable) -> Any:
        def nested_decorated_method_wrapper(self: Any, *args, **kwargs):
            return f(getattr(self, attr_name), *args, **kwargs)

        return nested_decorated_method_wrapper
