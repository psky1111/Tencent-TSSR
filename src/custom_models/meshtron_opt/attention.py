import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Union, Tuple, List
try:
    from flash_attn_interface import flash_attn_func
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FLASH_ATTN_3_AVAILABLE = False


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        use_flash: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)
        
        self.norm_q, self.norm_k = None, None
        if qk_norm:
            dim_head = self.width // self.heads
            eps = 1e-6
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)

    def forward(self, x):
        x = self.c_qkv(x)
        if self.norm_q is not None and self.norm_k is not None:
            qkv = rearrange(x, 'b n (head_num kind_num head_dim) -> kind_num b n head_num head_dim', head_num=self.heads, kind_num=3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q_norm = self.norm_q(q)
            k_norm = self.norm_k(k)
            qkv_norm = torch.stack([q_norm, k_norm, v], dim=0)
            x = rearrange(qkv_norm, 'kind_num b n head_num head_dim -> b n (head_num kind_num head_dim)')
  
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.use_flash:
            if FLASH_ATTN_3_AVAILABLE:
                out = flash_attn_func(q, k, v)[0].reshape(bs, n_ctx, -1)
            else:
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        use_flash: bool = False,
        qk_norm=False,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            qk_norm=qk_norm,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return self._forward(x)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)
        
        self.norm_q, self.norm_k = None, None
        if qk_norm:
            dim_head = self.width // self.heads
            eps = 1e-6
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        if self.norm_q is not None and self.norm_k is not None:
            q = rearrange(x, 'b n (head_num head_dim) -> b n head_num head_dim', head_num=self.heads)
            q_norm = self.norm_q(q)
            x = rearrange(q_norm, 'b n head_num head_dim -> b n (head_num head_dim)')
            
            kv = rearrange(data, 'b n (head_num kind_num head_dim) -> kind_num b n head_num head_dim', head_num=self.heads, kind_num=2)
            k, v = kv[0], kv[1]
            k_norm = self.norm_k(k)
            kv_norm = torch.stack([k_norm, v], dim=0)
            data = rearrange(kv_norm, 'kind_num b n head_num head_dim -> b n (head_num kind_num head_dim)')
        
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, use_flash: bool = False, n_data: Optional[int] = None):

        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        if self.use_flash:
            if FLASH_ATTN_3_AVAILABLE:
                out = flash_attn_func(q, k, v)[0].reshape(bs, n_ctx, -1)
            else:
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False,
        qk_norm=False,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            qk_norm=qk_norm,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x
