from jax import jit, vmap
from functools import partial
from training import layers


@partial(jit, static_argnums=2)
def forward(x, params, structure):
    residual = x
    for i in range(len(structure)):
        layer = structure[i][0]
        if layer == "attention":
            x = layers.attention_block(x, params[i])
        elif layer == "linear":
            x = layers.linear(x, params[i])
        elif layer == "layer_norm":
            x = layers.layer_norm(x, params[i])
        elif layer == "relu":
            x = layers.relu(x)
        elif layer == "sigmoid":
            x = layers.sigmoid(x)
        elif layer == "softmax":
            x = layers.softmax(x)
        elif layer == "sum_pool":
            x = layers.sum_pool(x)
        elif layer == "mean_pool":
            x = layers.mean_pool(x)
        elif layer == "collect_residual":
            residual = x
        elif layer == "add_residual":
            x = x + residual
        elif layer == "positional_embedding":
            x = x + params[i]
        else:
            raise Exception("Unidentified layer " + layer)

    return x


@partial(jit, static_argnums=2)
@partial(vmap, in_axes=(0, None, None))
def forward_ds(x, params, structure):
    return forward(x, params, structure)
