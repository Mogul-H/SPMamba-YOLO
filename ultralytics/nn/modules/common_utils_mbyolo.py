import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    # import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def _cross_scan_onnx(x: torch.Tensor):
    """Mutation-free CrossScan used only while tracing an ONNX graph."""
    x_hw = x.flatten(2, 3)
    x_wh = x.transpose(2, 3).contiguous().flatten(2, 3)
    x_forward = torch.stack((x_hw, x_wh), dim=1)
    return torch.cat((x_forward, torch.flip(x_forward, dims=[-1])), dim=1)


def _cross_merge_onnx(ys: torch.Tensor, H: int, W: int):
    """Mutation-free CrossMerge used only while tracing an ONNX graph."""
    B, K, D = ys.shape[:3]
    ys = ys.reshape(B, K, D, H * W)
    ys = ys[:, 0:2] + torch.flip(ys[:, 2:4], dims=[-1])
    y_hw = ys[:, 0]
    y_wh = ys[:, 1].reshape(B, D, W, H).transpose(2, 3).contiguous().reshape(B, D, H * W)
    return y_hw + y_wh


def _selective_scan_onnx(
        us: torch.Tensor,
        dts: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        Cs: torch.Tensor,
        Ds: torch.Tensor = None,
        delta_bias: torch.Tensor = None,
        delta_softplus: bool = True,
        chunksize: int = 256,
):
    """Pure-PyTorch, fixed-shape selective scan for ONNX tracing.

    The CUDA extension is opaque to the legacy PyTorch 2.1 ONNX tracer.  This
    equivalent chunked recurrence uses only standard tensor operators.  The
    Python loop is intentionally unrolled for the fixed 640x640 deployment
    graph.  FP32 is retained inside the state recurrence for stability.

    Each chunk uses an associative prefix scan.  This avoids both a 25,600-step
    ONNX loop and the exp/cumsum/divide formulation's underflow division.
    """
    input_dtype = us.dtype
    has_D = Ds is not None

    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.reshape(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)

    if Bs.dim() == 3:
        Bs = Bs.unsqueeze(1)
    if Cs.dim() == 3:
        Cs = Cs.unsqueeze(1)

    batch, groups, state_dim, length = Bs.shape
    us = us.reshape(batch, groups, -1, length).float()
    dts = dts.reshape(batch, groups, -1, length).float()
    As = As.reshape(groups, -1, state_dim).float()
    Bs = Bs.float()
    Cs = Cs.float()
    Ds = Ds.reshape(groups, -1).float() if has_D else None

    channels_per_group = As.shape[1]
    hprefix = us.new_zeros((batch, groups, channels_per_group, state_dim), dtype=torch.float32)
    output_chunks = []

    for start in range(0, length, chunksize):
        stop = min(start + chunksize, length)
        us_chunk = us[..., start:stop]
        dts_chunk = dts[..., start:stop]
        Bs_chunk = Bs[..., start:stop].permute(0, 1, 3, 2)
        Cs_chunk = Cs[..., start:stop].permute(0, 1, 3, 2)

        # State recurrence: h_t = a_t * h_(t-1) + b_t.
        # Shapes: a/b/state = (B, G, D, T, N).
        a = torch.exp(dts_chunk.unsqueeze(-1) * As.unsqueeze(0).unsqueeze(3))
        b = (dts_chunk * us_chunk).unsqueeze(-1) * Bs_chunk.unsqueeze(2)

        # Inclusive associative scan over T using pair composition:
        # (a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1).
        offset = 1
        chunk_length = stop - start
        while offset < chunk_length:
            a_previous = torch.cat((torch.ones_like(a[:, :, :, :offset]), a[:, :, :, :-offset]), dim=3)
            b_previous = torch.cat((torch.zeros_like(b[:, :, :, :offset]), b[:, :, :, :-offset]), dim=3)
            next_b = b + a * b_previous
            next_a = a * a_previous
            a, b = next_a, next_b
            offset *= 2

        states = b + a * hprefix.unsqueeze(3)
        output_chunks.append((states * Cs_chunk.unsqueeze(2)).sum(dim=-1))
        hprefix = states[:, :, :, -1, :]

    ys = torch.cat(output_chunks, dim=-1)
    if has_D:
        ys = ys + Ds.reshape(1, groups, -1, 1) * us
    ys = ys.reshape(batch, -1, length)
    return ys.to(input_dtype)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    exporting_onnx = torch.onnx.is_in_onnx_export()
    xs = _cross_scan_onnx(x) if exporting_onnx else CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    if exporting_onnx:
        ys = _selective_scan_onnx(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)
    else:
        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)
    ys = ys.view(B, K, -1, H, W)

    y: torch.Tensor = _cross_merge_onnx(ys, H, W) if exporting_onnx else CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)
