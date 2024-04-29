from .HardNegativeNLLLoss import HardNegativeNLLLoss

def load_loss(loss_class, *args, **kwargs):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
