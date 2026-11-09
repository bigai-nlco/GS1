"""Keep GDN normalization at the native single-branch kernel geometry."""

import torch


def l2norm_rows(value):
    from fla.modules.l2norm import l2norm_fwd

    # FLA selects its Triton reduction tile from the flattened row count.
    # Including the batch dimension can change rounding before the BF16 store.
    # The enclosing FLA scan normally provides this device guard.
    with torch.cuda.device(value.device):
        rows = [l2norm_fwd(row.contiguous())[0] for row in value.split(1)]
    return rows[0] if len(rows) == 1 else torch.cat(rows, 0)


def gated_norm_rows(module, value, gate):
    width = value.shape[-1]
    rows = [
        module(row.reshape(-1, width), z.reshape(-1, width))
        for row, z in zip(value.split(1), gate.split(1), strict=True)
    ]
    return (rows[0] if len(rows) == 1 else torch.cat(rows, 0)).reshape(value.shape)
