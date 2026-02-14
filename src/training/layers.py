import jax.numpy as jnp
import jax.random as jrd

from jax import jit



@jit
def attention_block(x, params):
    """
    x.shape = (B, L, d_model)
    W_Q.shape = (d_model, d_QK)
    W_K.shape = (d_model, d_QK)
    W_V.shape = (d_model, d_V)
    """

    W_Q, W_K, W_V = params

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    A = Q @ jnp.swapaxes(K, -1, -2)
    A = A / jnp.sqrt(Q.shape[-1])
    A = softmax(A)

    y = A @ V
    return y


@jit
def linear(x, params):
    """
    x.shape = (B, L, d_in)
    W.shape = (d_in, d_out)
    b.shape = (d_out,)
    """

    W, b = params
    y = x @ W + b
    return y


@jit
def layer_norm(x, params):
    gamma, beta = params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + 1e-6)
    y = gamma * x_hat + beta
    return y


@jit
def dropout(x, p, subkey):
    mask = jrd.uniform(subkey, shape=x.shape)
    mask = (mask > p).astype(jnp.int32)
    return x * mask / (1 - p)


@jit
def softmax(x, axis=-1):
    e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)


@jit
def relu(x):
    return jnp.maximum(0, x)


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jit
def sum_pool(x):
    return jnp.sum(x, axis=-2)


@jit
def mean_pool(x):
    return jnp.mean(x, axis=-2)

