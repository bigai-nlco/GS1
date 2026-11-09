"""GS1: native typed decision models and optimized inference."""

__version__ = "0.1.0"
__all__ = ["Engine", "GS1Config", "GS1Model", "DecisionRequest", "DecisionResponse"]


def __getattr__(name):
    if name == "Engine":
        from .inference.engine import Engine

        return Engine
    if name in {"GS1Config", "GS1Model"}:
        from .model import modular_gs1

        return getattr(modular_gs1, name)
    if name in {"DecisionRequest", "DecisionResponse"}:
        from .model import schema

        return getattr(schema, name)
    raise AttributeError(name)
