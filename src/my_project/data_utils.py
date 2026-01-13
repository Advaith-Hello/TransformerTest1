from jax import Array
from jax import jit
from functools import partial



@partial(jit, static_argnums=2)
def batch(
        x: Array,
        y: Array,
        batch_size: int,
    ) -> tuple[Array, Array]:

    """
    x: (N, H, W, C)
    y: (N, )
    returns:
      x: (num_batches, batch_size, H, W, C)
      y: (num_batches, batch_size)
    """

    num_batches = x.shape[0] // batch_size
    x = x[:num_batches * batch_size]
    y = y[:num_batches * batch_size]

    x = x.reshape(
        num_batches,
        batch_size,
        x.shape[1], # H
        x.shape[2], # W
        x.shape[3], # C
    )
    y = y.reshape(num_batches, batch_size)

    return x, y
