"""Offline Tkinter UI; importing data services does not initialize the GUI."""
from importlib import import_module
__all__ = ["AnviksaApp", "main"]

def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".app", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
