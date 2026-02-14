import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

from functools import partial
from jax import jit



@partial(jit, static_argnums=1)
def images_to_patches(x, patch_size_dim, key):
    """
    x: (N, H, W, C)
    returns: (N, num_patches, patch_size)
    """

    N, H, W, C = x.shape

    num_patches_per_dim = H // patch_size_dim
    num_patches = num_patches_per_dim ** 2
    patch_size = C * (patch_size_dim ** 2)

    x = x.reshape(
        N,
        num_patches_per_dim,
        patch_size_dim,
        num_patches_per_dim,
        patch_size_dim,
        C
    )

    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(N, num_patches, patch_size)

    return x, key


@partial(jit, static_argnums=1)
def pad_to(x, resize_shape, key):
    H, W = x.shape[1], x.shape[2]

    pad_x = max(resize_shape[0] - H, 0)
    pad_y = max(resize_shape[1] - W, 0)

    pad_x_1 = pad_x // 2
    pad_x_2 = pad_x - pad_x_1
    pad_y_1 = pad_y // 2
    pad_y_2 = pad_y - pad_y_1

    return jnp.pad(x, mode="constant", pad_width=(
        (0, 0),
        (pad_x_1, pad_x_2),
        (pad_y_1, pad_y_2),
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
        slice_sizes=(x.shape[0], *crop_shape, x.shape[3]),
    ), key


@partial(jit, static_argnums=1)
def vertical_flip(x, p, key):
    key, subkey = jrd.split(key, num=2)
    return lax.cond(
        jrd.uniform(subkey) < p,
        lambda x: jnp.flip(x, axis=1),
        lambda x: x,
        operand=x
    ), key


@partial(jit, static_argnums=1)
def horizontal_flip(x, p, key):
    key, subkey = jrd.split(key, num=2)
    return lax.cond(
        jrd.uniform(subkey) < p,
        lambda x: jnp.flip(x, axis=2),
        lambda x: x,
        operand=x
    ), key


@partial(jit, static_argnums=1)
def cutout(x, cutout_proportion, key):
    key, k1, k2 = jrd.split(key, num=3)

    cut_h, cut_w = (
        int(cutout_proportion * x.shape[1]),
        int(cutout_proportion * x.shape[2]),
    )

    y_start = jrd.randint(k1, (), minval=0, maxval=(x.shape[1] - cut_h + 1))
    x_start = jrd.randint(k2, (), minval=0, maxval=(x.shape[2] - cut_w + 1))

    return lax.dynamic_update_slice(
        x,
        update=jnp.zeros(shape=(x.shape[0], cut_h, cut_w, x.shape[3]), dtype=x.dtype),
        start_indices=(0, y_start, x_start, 0),
    ), key
