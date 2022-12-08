import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import profile_nn_jax


def print_and_return_zero(
    message,
    timeit,
    shapes,
    dtypes,
    mean=None,
    amplitude=None,
    minval=None,
    maxval=None,
    hasnan=None,
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
            t = f"  {1e6 * t: 6.1f}us"
    else:
        t = ""

    flags = []
    if hasnan:
        flags += ["NaN"]
    if any(d == np.float16 for d in dtypes):
        flags += ["f16"]
    if any(d == np.float32 for d in dtypes):
        flags += ["f32"]
    if any(d == np.float64 for d in dtypes):
        flags += ["f64"]
    if any(d == np.int8 for d in dtypes):
        flags += ["i8"]
    if any(d == np.int16 for d in dtypes):
        flags += ["i16"]
    if any(d == np.int32 for d in dtypes):
        flags += ["i32"]
    if any(d == np.int64 for d in dtypes):
        flags += ["i64"]
    if any(d == np.uint8 for d in dtypes):
        flags += ["u8"]
    if any(d == np.uint16 for d in dtypes):
        flags += ["u16"]
    if any(d == np.uint32 for d in dtypes):
        flags += ["u32"]
    if any(d == np.uint64 for d in dtypes):
        flags += ["u64"]
    if any(d == np.bool_ for d in dtypes):
        flags += ["bool"]
    if any(d == np.complex64 for d in dtypes):
        flags += ["c64"]
    if any(d == np.complex128 for d in dtypes):
        flags += ["c128"]

    if len(shapes) == 1:
        s = f"{shapes[0]}"
    else:
        s = f"{shapes}"

    total_len = 35
    i = total_len - len(message)

    msg = f"{'-' * (i//2)} {message[:total_len]} {'-' * (i - i//2)}{s}{t} "
    if mean is not None:
        msg += f" {mean: 8.1e} ±{amplitude: 8.1e} [{minval: 7.1e},{maxval: 7.1e}] {','.join(flags)}"

    logging.info(msg)

    return np.array(0, dtype=np.int32)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def profile(message: str, x):
    if profile_nn_jax.is_enabled():
        leaves = jax.tree_util.tree_leaves(x)
        if hasattr(x, "shape"):
            shapes = [x.shape]
        else:
            shapes = [e.shape for e in leaves]

        dtypes = [e.dtype for e in leaves]

        if profile_nn_jax.is_timing():
            fn = partial(
                jax.pure_callback,
                callback=partial(print_and_return_zero, message, True, shapes, dtypes),
                result_shape_dtypes=jnp.array(0, dtype=jnp.int32),
            )
        else:
            fn = partial(
                jax.debug.callback,
                callback=partial(print_and_return_zero, message, False, shapes, dtypes),
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
