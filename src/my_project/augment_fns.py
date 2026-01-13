import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

from functools import partial
from jax import jit



@partial(jit, static_argnums=1)
def pad_to(x, resize_shape, key):
    resize_pad_x = jnp.maximum(resize_shape[0] - x.shape[1], 0)
    resize_pad_y = jnp.maximum(resize_shape[1] - x.shape[2], 0)

    resize_pad_x_1 = resize_pad_x // 2
    resize_pad_x_2 = resize_pad_x - resize_pad_x_1
    resize_pad_y_1 = resize_pad_y // 2
    resize_pad_y_2 = resize_pad_y - resize_pad_y_1

    return jnp.pad(x, mode="constant", pad_width=(
        (0, 0),
        (resize_pad_x_1, resize_pad_x_2),
        (resize_pad_y_1, resize_pad_y_2),
        (0, 0),
    )), key


@partial(jit, static_argnums=1)
def crop_to(x, crop_shape, key):
    key, k1, k2 = jrd.split(key, num=3)
    start_indices = (
        0,
        jrd.randint(k1, shape=(), minval=0, maxval=x.shape[1] - crop_shape[0] + 1),
        jrd.randint(k2, shape=(), minval=0, maxval=x.shape[2] - crop_shape[1] + 1),
        0,
    )
    return lax.dynamic_slice(
        x,
        start_indices=start_indices,
        slice_sizes=crop_shape,
    ), key


@partial(jit, static_argnums=1)
def vertical_flip(x, p, key):
    key, subkey = jrd.split(key, num=2)
    return (
        jnp.flip(x, axis=1)
        if jrd.uniform(subkey) < p
        else x
    ), key


@partial(jit, static_argnums=1)
def horizontal_flip(x, p, key):
    key, subkey = jrd.split(key, num=2)
    return (
        jnp.flip(x, axis=2)
        if jrd.uniform(subkey) < p
        else x
    ), key


@partial(jit, static_argnums=1)
def cutout(x, cutout_proportion, key):
    key, k1, k2 = jrd.split(key, num=3)

    cut_h, cut_w = (
        jnp.int32(cutout_proportion * x.shape[1]),
        jnp.int32(cutout_proportion * x.shape[2]),
    )

    y_start = jrd.randint(k1, (), minval=0, maxval=(x.shape[1] - cut_h + 1))
    x_start = jrd.randint(k2, (), minval=0, maxval=(x.shape[2] - cut_w + 1))

    cutout_patch = jnp.zeros(shape=(
        x.shape[0],
        cut_h,
        cut_w,
        x.shape[3]
    ), dtype=x.dtype)

    return lax.dynamic_update_slice(
        x,
        update=cutout_patch,
        start_indices=(0, y_start, x_start, 0),
    ), key
