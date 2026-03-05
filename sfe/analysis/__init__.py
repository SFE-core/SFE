__all__ = ["strain", "finance", "eeg", "shm", "regimes"]

def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f".{name}", __package__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")