from functools import partial
from jax import jit, vmap


@partial(jit, static_argnums=2)
def augment(x, key, augments):
    for fn, arg in augments:
        x, key = fn(x, arg, key)

    return x, key


@partial(jit, static_argnums=2)
@partial(vmap, in_axes=(0, None, None))
def augment_ds(x, key, augments):
    return augment(x, key, augments)
