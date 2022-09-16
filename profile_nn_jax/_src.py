from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import profile_nn_jax


def print_and_return_zero(message, timeit, mean, amplitude, minval, maxval, hasnan):
    if timeit:
        t = profile_nn_jax.restart_timer()

        if t is None:
            t = " " + "*" * 7
        else:
            t = f" {t:06.3f}s"
    else:
        t = ""

    flags = []
    if hasnan:
        flags += ["NaN"]

    i = 20 - len(message)
    print(
        f"{'-' * (i//2)} {message[:20]} {'-' * (i - i//2)}{t}{mean: 8.1e} ±{amplitude: 8.1e} [{minval: 6.0e},{maxval: 6.0e}] {','.join(flags)}",
        flush=True,
    )

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

        zero = fn(
            mean=jnp.mean(jnp.array([e.mean() for e in leaves])),
            amplitude=jnp.mean(jnp.array([(e**2).mean() for e in leaves])) ** 0.5,
            minval=jnp.min(jnp.array([e.min() for e in leaves])),
            maxval=jnp.max(jnp.array([e.max() for e in leaves])),
            hasnan=jnp.any(jnp.array([jnp.isnan(e).any() for e in leaves])),
        )
        if isinstance(zero, jnp.ndarray):
            return jax.tree_util.tree_map(lambda e: e + zero, x)
    return x


@profile.defjvp
def profile_jvp(message, primals, tangents):
    (x,) = primals
    (dx,) = tangents
    return profile(f"(jvp){message}", x), dx
