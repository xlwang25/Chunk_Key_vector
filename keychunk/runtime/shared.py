"""Model and tensor helpers used by the local GPU backend."""

def resolve_layers(model):
    if hasattr(model,'layers'):return model.layers
    if hasattr(model,'model') and hasattr(model.model,'layers'):return model.model.layers
    raise RuntimeError('cannot resolve decoder layers')
