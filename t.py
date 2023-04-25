import jax
import jax.numpy as jnp
import profile_nn_jax


@jax.jit
def f(x):
    x = profile_nn_jax.profile("x", x)
    x = jnp.sin(x)
    x = profile_nn_jax.profile("sin(x)", x)
    return x


profile_nn_jax.enable()  # Enable profiling before the first call to f() (before the first compilation)
f(jnp.arange(10))

jax.grad(lambda x: jnp.sum(f(x)))(jnp.arange(10.0))
