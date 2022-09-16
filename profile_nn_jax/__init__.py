from time import perf_counter as _perf_counter
from typing import Optional

from ._src import profile

__version__ = "0.1.0"

ENABLED = False


def enable(yes: bool = True):
    global ENABLED
    ENABLED = yes


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


__all__ = ["profile"]
