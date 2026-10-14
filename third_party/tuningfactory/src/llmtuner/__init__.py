"""Pinned training implementation with lazy optional entry points."""
__version__ = "0.5.3"


def run_exp(*args, **kwargs):
    from .train.tuner import run_exp as implementation
    return implementation(*args, **kwargs)


def export_model(*args, **kwargs):
    from .train.tuner import export_model as implementation
    return implementation(*args, **kwargs)
