"""
Sample models for usage
Use training.parameterize.parameterize() to init params
Use training.forward.forward() to run the model

linear: (d_in, d_out)
    Does W @ x + b

attention: (W_QK, W_V, token_length)
    Init takes length of weights Q, K, V and token_length
    Implementation of a standard single-head attention mechanism

layer_norm: (token_length,)
    Normalizes mean to 0 and std to 1 over batches

positional_embedding: (N_tokens, token_length)
    Adds a leaned positional embedding per batch

collect_residual: ()
    Stores value of x at the given position
    By default this value is the input to the forward function

add_residual: ()
    Adds the residual collected by collect_residual to x

relu: ()
    Activation function
    Replaces all negative numbers with 0
    x = max(0, x)

sigmoid: ()
    Activation function
    Forces all numbers between 0 and 1
    x = 1 / (1 + exp(-x))

mean_pool: ()
    Makes a 2d batch 1d
    Takes the mean of each embedding dimension

sum_pool: ()
    Makes a 2d batch 1d
    Takes the sum of each embedding dimension
"""


model1 = (
    ("linear", (16, 64)),
    ("positional_embedding", (49, 64)),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("linear", (64, 128)),
    ("relu", ()),
    ("linear", (128, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("linear", (64, 128)),
    ("relu", ()),
    ("linear", (128, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("mean_pool", ()),
    ("linear", (64, 10)),
)


model2 = (
    ("linear", (16, 64)),
    ("positional_embedding", (49, 64)),

    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("linear", (64, 256)),
    ("relu", ()),
    ("linear", (256, 64)),
    ("add_residual", ()),

    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("linear", (64, 256)),
    ("relu", ()),
    ("linear", (256, 64)),
    ("add_residual", ()),

    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("linear", (64, 256)),
    ("relu", ()),
    ("linear", (256, 64)),
    ("add_residual", ()),

    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),
    ("collect_residual", ()),
    ("linear", (64, 256)),
    ("relu", ()),
    ("linear", (256, 64)),
    ("add_residual", ()),

    ("layer_norm", (64,)),
    ("mean_pool", ()),
    ("linear", (64, 10)),
)
