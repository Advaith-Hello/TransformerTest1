from functools import partial
from jax import jit


@partial(jit, static_argnums=2)
def augment(x, key, augments):
    for fn, arg in augments:
        x, key = fn(x, arg, key)

    return x, key
