"""Immutable, inference-only GS1 recurrent state and tensor-tree operations."""

from dataclasses import fields

import torch

from gs1.model.modular_gs1 import GDNState


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, GDNState):
        return GDNState(**{f.name: map_tensors(getattr(value, f.name), function) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    return value


def tensors(value):
    result = []
    map_tensors(value, lambda tensor: result.append(tensor))
    return result


def tensor_signature(value, strides=True):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()) if strides else None, str(value.dtype))
    if isinstance(value, GDNState):
        return (
            "gdn",
            tuple((f.name, tensor_signature(getattr(value, f.name), strides)) for f in fields(value)),
        )
    if isinstance(value, tuple):
        return tuple(tensor_signature(item, strides) for item in value)
    if isinstance(value, dict):
        return tuple((key, tensor_signature(item, strides)) for key, item in value.items())
    return value


def clone_state(value):
    copies = {}

    def clone(tensor):
        if id(tensor) not in copies:
            copies[id(tensor)] = tensor.detach().clone()
        return copies[id(tensor)]

    return map_tensors(value, clone)


def state_bytes(value):
    return sum(
        {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors(value)}.values()
    )


def state_row(value, row):
    return map_tensors(value, lambda tensor: tensor[row : row + 1])


def batch_states(values):
    first = values[0]
    if isinstance(first, GDNState):
        if any(value.position != first.position for value in values):
            raise ValueError("GDN cohorts must have the same absolute prefix length")
        return GDNState(**{f.name: batch_states([getattr(v, f.name) for v in values]) for f in fields(first)})
    if isinstance(first, torch.Tensor):
        if any(value.shape[1:] != first.shape[1:] for value in values):
            raise ValueError("Recurrent cohorts must have identical state geometry")
        return torch.cat(values, dim=0)
    if isinstance(first, tuple):
        return tuple(batch_states([value[index] for value in values]) for index in range(len(first)))
    if any(value != first for value in values):
        raise ValueError("Incompatible recurrent cohort metadata")
    return first


def parameter_fingerprint(model):
    # Respect the model's native traversal, including custom parameter iterators.
    return tuple((p.data_ptr(), p._version, str(p.dtype)) for p in model.parameters()) + tuple(
        (b.data_ptr(), b._version, str(b.dtype)) for b in model.buffers()
    )


def branch_geometry(rows):
    return tuple(
        (
            len(state),
            len(b.suffix),
            b.token_roles,
            b.logical_positions,
            b.question_start_position,
            b.question_span,
            b.question_end_position,
            b.option_starts,
            b.option_spans,
            b.option_ends,
            b.decision_position,
            b.state_evidence_start,
            len(b.candidates),
        )
        for state, b in rows
    )
