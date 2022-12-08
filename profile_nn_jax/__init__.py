from time import perf_counter as _perf_counter
from typing import Optional

__version__ = "0.2.0"

ENABLED = False
TIMIMNG = False
STATISTICS = False


def enable(*, timing: bool = True, statistics: bool = True):
    import logging

    global ENABLED
    global TIMIMNG
    global STATISTICS
    ENABLED = True
    TIMIMNG = timing
    STATISTICS = statistics
    logging.getLogger().setLevel(logging.INFO)


def disable():
    global ENABLED
    ENABLED = False


def is_enabled() -> bool:
    return ENABLED


def is_timing() -> bool:
    return TIMIMNG


def is_statistics() -> bool:
    return STATISTICS


REF_TIME = None


def start_timer():
    global REF_TIME
    REF_TIME = _perf_counter()


def elapsed_time() -> Optional[float]:
    global REF_TIME
    if REF_TIME is None:
        return None
    else:
        return _perf_counter() - REF_TIME


def stop_timer() -> Optional[float]:
    global REF_TIME
    t = elapsed_time()
    REF_TIME = None
    return t


def restart_timer() -> Optional[float]:
    t = stop_timer()
    start_timer()
    return t


def profile(message: str, x):
    """Add a profile message to the computation graph.

    Args:
        message (str): The message to print.
        x (pytree): A variable to bind the message to.

    Returns:
        The same variable as `x`.

    Example:
        >>> import jax.numpy as jnp
        >>> from profile_nn_jax import profile
        >>> x = jnp.arange(10)
        >>> x = profile("x", x)
    """
    from ._src import profile as _profile

    return _profile(message, x)
