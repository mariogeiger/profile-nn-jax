import jax
import jax.numpy as jnp

import profile_nn_jax


@jax.jit
def f(x):
    x = profile_nn_jax.profile("x", x)
    x = jnp.sin(x)
    x = profile_nn_jax.profile("sin(x)", x)
    x = jnp.cos(x)
    x = profile_nn_jax.profile("cos(x)", x)
    y = jnp.sum(x)
    y = profile_nn_jax.profile("sum(x)", y)
    return x


def test():
    profile_nn_jax.enable()
    x = jnp.arange(10)
    f(x)

    profile_nn_jax.stop_timer()
    f(x)


if __name__ == "__main__":
    test()
