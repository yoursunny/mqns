from typing import cast


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
