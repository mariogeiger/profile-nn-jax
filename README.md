# Profile NN jax

```python
import profile_nn_jax
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    x = profile_nn_jax.profile("x", x)
    x = jnp.sin(x)
    x = profile_nn_jax.profile("sin(x)", x)
    return x


profile_nn_jax.enable()  # disabled by default
f(jnp.arange(10))
```
