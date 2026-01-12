import jax
import jax.numpy as jnp
import optax

from jax import jit
from my_project import forward


def cross_entropy_loss_raw_python(x, y, params, structure):
    logits = forward.forward(x, params, structure)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)


def mse_loss_raw_python(x, y, params, structure):
    logits = forward.forward(x, params, structure)
    loss = (logits - y) ** 2
    return jnp.mean(loss)


cross_entropy_loss = jit(cross_entropy_loss_raw_python, static_argnums=3)

cross_entropy_loss_value_and_grad = jit(
    jax.value_and_grad(
        cross_entropy_loss_raw_python,
        argnums=2
    ), static_argnums=3
)

mse_loss = jit(mse_loss_raw_python, static_argnums=3)

mse_loss_value_and_grad = jit(
    jax.value_and_grad(
        mse_loss_raw_python,
        argnums=2
    ), static_argnums=3
)
