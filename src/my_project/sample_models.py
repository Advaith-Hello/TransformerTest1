"""
Sample models for usage
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
