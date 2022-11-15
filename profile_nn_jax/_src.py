import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import profile_nn_jax


def print_and_return_zero(
    message, timeit, mean=None, amplitude=None, minval=None, maxval=None, hasnan=None
):
    if timeit:
        t = profile_nn_jax.restart_timer()

        if t is None:
            t = " " + "*" * 9
        elif t > 1.0:
            t = f" {t: 5.1f}s   "
        elif t > 1e-3:
            t = f"  {1000 * t: 5.1f}ms "
        else:
            t = f"   {1e6 * t: 5.1f}us"
    else:
        t = ""

    flags = []
    if hasnan:
        flags += ["NaN"]

    total_len = 35
    i = total_len - len(message)

    msg = f"{'-' * (i//2)} {message[:total_len]} {'-' * (i - i//2)}{t}"
    if mean is not None:
        msg += f" {mean: 8.1e} ±{amplitude: 8.1e} [{minval: 7.1e},{maxval: 7.1e}] {','.join(flags)}"

    logging.info(msg)

    return np.array(0, dtype=np.int32)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def profile(message: str, x):
    if profile_nn_jax.is_enabled():
        leaves = jax.tree_util.tree_leaves(x)
        if profile_nn_jax.is_timing():
            fn = partial(
                jax.pure_callback,
                callback=partial(print_and_return_zero, message, True),
                result_shape_dtypes=jnp.array(0, dtype=jnp.int32),
            )
        else:
            fn = partial(
                jax.debug.callback,
                callback=partial(print_and_return_zero, message, False),
            )

        if profile_nn_jax.is_statistics():
            zero = fn(
                mean=jnp.mean(jnp.array([e.mean() for e in leaves])),
                amplitude=jnp.mean(jnp.array([(e**2).mean() for e in leaves])) ** 0.5,
                minval=jnp.min(jnp.array([e.min() for e in leaves])),
                maxval=jnp.max(jnp.array([e.max() for e in leaves])),
                hasnan=jnp.any(jnp.array([jnp.isnan(e).any() for e in leaves])),
            )
        else:
            zero = fn()

        if isinstance(zero, jnp.ndarray):
            return jax.tree_util.tree_map(lambda e: e + zero, x)
    return x


@profile.defjvp
def profile_jvp(message, primals, tangents):
    (x,) = primals
    (dx,) = tangents
    return profile(f"(jvp){message}", x), dx
