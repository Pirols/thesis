from src.optimizers.lookahead import Lookahead
from src.optimizers.ralamb import Ralamb


def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):  # noqa: N802
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)
