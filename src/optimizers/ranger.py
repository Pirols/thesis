from torch.optim import RAdam

from src.optimizers.lookahead import Lookahead


def Ranger(params, alpha=0.5, k=6, *args, **kwargs):  # noqa: N802
    radam = RAdam(params, *args, **kwargs)
    return Lookahead(radam, alpha, k)
