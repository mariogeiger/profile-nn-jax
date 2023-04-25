# Profile NN jax

Install from github:

```bash
pip install git+https://github.com/mariogeiger/profile-nn-jax.git
```

## Example usage

```python
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
```

## Example output

```bash
INFO:root:----------------- x -----------------(10,) *********   4.5e+00 ± 5.3e+00 [ 0.0e+00, 9.0e+00] i32
INFO:root:-------------- sin(x) ---------------(10,)   325.2us   2.0e-01 ± 6.9e-01 [-9.6e-01, 9.9e-01] f32
```
