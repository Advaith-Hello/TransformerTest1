from jax import Array
from jax import jit
from functools import partial


@partial(jit, static_argnums=1)
def images_to_patches(x: Array, patch_size_dim: int) -> Array:
    """
    x: (N, H, W)
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

    return x


@partial(jit, static_argnums=(2, 3))
def batch_and_patch(
        x: Array,
        y: Array,
        batch_size: int,
        patch_size_dim: int,
    ) -> tuple[Array, Array]:

    """
    x: (N, H, W, C)
    y: (N, )
    returns:
      x: (num_batches, batch_size, num_patches, patch_size)
      y: (num_batches, batch_size)
    """

    num_batches = x.shape[0] // batch_size
    x = x[:num_batches * batch_size]
    y = y[:num_batches * batch_size]

    x = images_to_patches(x, patch_size_dim)
    x = x.reshape(
        num_batches,
        batch_size,
        x.shape[1],  # num_patches
        x.shape[2],  # patch_size
    )
    y = y.reshape(num_batches, batch_size)

    return x, y

