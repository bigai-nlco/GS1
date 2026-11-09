"""The GS1 native decision model. Model imports are loaded lazily."""

__all__ = ["GS1Config", "GS1PretrainedModel", "GS1TextModel", "GS1Model"]


def __getattr__(name):
    if name in __all__:
        from gs1.model import modular_gs1

        return getattr(modular_gs1, name)
    raise AttributeError(name)
