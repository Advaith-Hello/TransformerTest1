import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

from functools import partial
from jax import jit



@partial(jit, static_argnums=1)
def pad_to(x, resize_shape, key):
    resize_pad_1 = resize_shape - x.shape // 2
    resize_pad_2 = resize_shape - x.shape - resize_pad_1
    return jnp.pad(x, mode="empty", pad_width=(
        (0, 0),
        (resize_pad_1, resize_pad_2),
        (resize_pad_1, resize_pad_2),
        (0, 0),
    )), key


@partial(jit, static_argnums=1)
def crop_to(x, crop_shape, key):
    key, k1, k2 = jrd.split(key, num=3)
    start_indices = (
        jrd.randint(k1, shape=(), minval=0, maxval=x.shape[0] - crop_shape[0] + 1),
        jrd.randint(k2, shape=(), minval=0, maxval=x.shape[1] - crop_shape[1] + 1),
    )
    return lax.dynamic_slice(
        x,
        start_indices=start_indices,
        slice_sizes=crop_shape,
    ), key
