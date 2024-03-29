from time import perf_counter as _perf_counter
from typing import Optional
import jax.numpy as jnp

__version__ = "0.2.1"

ENABLED = False


def enable():
    """Enable profiling."""
    global ENABLED
    ENABLED = True


def disable():
    global ENABLED
    ENABLED = False


def is_enabled() -> bool:
    return ENABLED


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


def profile(message: str, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
    """Add a profile message to the computation graph.

    Args:
        message (str): The message to print.
        x (pytree): A variable to bind the message to.
        mask (pytree): A mask to apply to the variable before computing the statistics.

    Returns:
        The same variable as `x`.

    Example:
        >>> import jax.numpy as jnp
        >>> from profile_nn_jax import profile
        >>> x = jnp.arange(10)
        >>> x = profile("x", x)
    """
    from ._src import profile as _profile

    return _profile(message, x, mask)
