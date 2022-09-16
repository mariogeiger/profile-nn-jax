import profile_nn_jax
import jax
import jax.numpy as jnp


@jax.jit
def f(x):
    x = profile_nn_jax.profile("x", x)
    x = jnp.sin(x)
    x = profile_nn_jax.profile("sin(x)", x)
    x = jnp.cos(x)
    x = profile_nn_jax.profile("cos(x)", x)
    return x


def test():
    profile_nn_jax.enable()
    x = jnp.arange(10)
    f(x)

    profile_nn_jax.stop_timer()
    f(x)


if __name__ == "__main__":
    test()
