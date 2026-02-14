import jax.numpy as jnp
import jax.random as jrd


def parameterize(structure, init_key=0) -> list:
    key = jrd.PRNGKey(init_key)
    params = []

    for layer, shape in structure:
        if layer == "attention":
            key, k1, k2, k3 = jrd.split(key, num=4)
            d_model, d_QK, d_V = shape
            params.append([
                jrd.normal(k1, shape=(d_model, d_QK)) * 0.02,
                jrd.normal(k2, shape=(d_model, d_QK)) * 0.02,
                jrd.normal(k3, shape=(d_model, d_V)) * 0.02,
            ])

        elif layer == "linear":
            key, subkey = jrd.split(key)
            # shape.shape = d_in, d_out
            params.append([
                jrd.normal(subkey, shape=shape) * 0.01,
                jnp.zeros(shape=(shape[1],))
            ])

        elif layer == "layer_norm":
            params.append([
                jnp.ones(shape=shape),
                jnp.zeros(shape=shape),
            ])

        elif layer == "positional_embedding":
            key, subkey = jrd.split(key)
            # shape.shape = (num_tokens, token_length)
            params.append(0.02 * jrd.normal(subkey, shape=shape))

        else:
            params.append([])

    return params
