def run(*args, **kwargs):
    from .generate import run as implementation
    return implementation(*args, **kwargs)

__all__ = ["run"]
