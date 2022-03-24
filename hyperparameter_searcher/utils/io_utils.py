"""Input-Output Utilities"""
from typing import Any


def fullname(cls: Any) -> str:
    """Function to return the full name of a particular class, used for hydra instantiate _target_"""
    module = cls.__module__
    if (
        module is None or module == str.__class__.__module__
    ):  # don't want to return 'builtins'
        return cls.__name__
    return module + "." + cls.__name__


def debug_function(x: float):
    """Debugging function"""
    return x**2
