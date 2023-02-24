import time
from functools import partial

import jax
import jax.numpy as jnp

import profile_nn_jax


@partial(jax.jit, static_argnums=(0,))
def f(enable_profiling: bool, x):
    if enable_profiling:
        profile_nn_jax.enable()
    else:
        profile_nn_jax.disable()

    x = profile_nn_jax.profile("x", x)
    x = jnp.sin(x)
    x = profile_nn_jax.profile("sin(x)", x)
    x = jnp.cos(x)
    x = profile_nn_jax.profile("cos(x)", x)

    return x


def test():
    x = jnp.arange(10)

    t = time.perf_counter()
    f(False, x)
    print(f"Profiling disabled: took {time.perf_counter() - t:.3f}s")

    t = time.perf_counter()
    f(True, x)
    print(f"Profiling enabled: took {time.perf_counter() - t:.3f}s")


if __name__ == "__main__":
    test()
