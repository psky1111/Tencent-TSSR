import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, Callable
from xformers.ops import fmha
try:
    from flash_attn_interface import flash_attn_func
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FLASH_ATTN_3_AVAILABLE = False

from src.custom_models.meshtron_opt.cache_utils import CustomSlidingCache


class MultiLevelRoPE(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        max_token_seq_len: int,
        theta_face_level: float = 10000.0,
        theta_hier_face: float = 10000.0,
        theta_hier_vertex: float = 10000.0,
        theta_hier_coord: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        assert max_token_seq_len % 9 == 0, "Max token length must be divisible by 9."
        
        self.dim = dim
        self.max_seq_len = max_token_seq_len
        
        # --- Pre-compute and register buffers ---
        
        # 1. Pre-compute for 'face' level
        freqs_face = self._create_freqs_cis(
            'face', max_token_seq_len // 9, dim, {'face_level': theta_face_level}
        )
        self.register_buffer("freqs_cis_face", freqs_face)
        
        # 2. Pre-compute for 'vertex' level
        freqs_vertex = self._create_freqs_cis(
            'vertex', max_token_seq_len // 3, dim, 
            {'hier_face': theta_hier_face, 'hier_vertex': theta_hier_vertex, 'hier_coord': theta_hier_coord}
        )
        self.register_buffer("freqs_cis_vertex", freqs_vertex)
        
        # 3. Pre-compute for 'token' level
        freqs_token = self._create_freqs_cis(
            'token', max_token_seq_len, dim, 
            {'hier_face': theta_hier_face, 'hier_vertex': theta_hier_vertex, 'hier_coord': theta_hier_coord}
        )
        self.register_buffer("freqs_cis_token", freqs_token)

    def _create_freqs_cis(self, level, seq_len, dim, thetas):
        """
        Helper function to compute the complex rotary embeddings during init.
        This version correctly replicates the original logic.
        """
        if level == 'face':
            positions = torch.arange(seq_len)
            freqs = 1.0 / (thetas['face_level'] ** (torch.arange(0, dim, 2).float() / dim))
            rotations = positions.float().unsqueeze(-1) * freqs
            
        elif level in ['token', 'vertex']:
            dim_third = (dim // 3) & ~1
            component_dims = [dim_third, dim_third, dim - 2 * dim_third]
            
            abs_positions = torch.arange(seq_len)
            
            # --- THE CRITICAL BUG FIX IS HERE ---
            # Replicate the scaling from the original leaky code
            if level == 'vertex':
                abs_positions = abs_positions * 3
            # --- END OF FIX ---
            
            face_pos = torch.div(abs_positions, 9, rounding_mode='floor')
            vertex_pos = torch.div(abs_positions, 3, rounding_mode='floor') % 3
            coord_pos = abs_positions % 3
            
            positions_list = [face_pos, vertex_pos, coord_pos]
            thetas_list = [thetas['hier_face'], thetas['hier_vertex'], thetas['hier_coord']]
            
            rotations_list = []
            for i, chunk_dim in enumerate(component_dims):
                if chunk_dim > 0:
                    freqs = 1.0 / (thetas_list[i] ** (torch.arange(0, chunk_dim, 2).float() / chunk_dim))
                    pos = positions_list[i]
                    rotations_list.append(pos.float().unsqueeze(-1) * freqs)
            
            rotations = torch.cat(rotations_list, dim=-1)
        
        freqs_cis = torch.polar(torch.ones_like(rotations), rotations)
        return freqs_cis

    def forward(self, level: str, seq_len: int, device: torch.device):
        """
        The main interface. Gets the correct RoPE tensor by SLICING.
        This no longer contains the "double division" bug.
        """
        if level == 'face':
            precomputed_freqs = self.freqs_cis_face
        elif level == 'vertex':
            precomputed_freqs = self.freqs_cis_vertex
        elif level == 'token':
            precomputed_freqs = self.freqs_cis_token
        else:
            raise NotImplementedError(f"Level '{level}' not supported.")

        if seq_len > precomputed_freqs.shape[0]:
            raise ValueError(f"Requested seq_len {seq_len} for level '{level}' is larger than pre-computed max {precomputed_freqs.shape[0]}")

        return precomputed_freqs[:seq_len].to(device)

    def apply(self, xq, xk, freqs_cis):
        # Your robust apply method remains unchanged and correct.
        freqs_cis_q, freqs_cis_k = freqs_cis
        orig_dtype = xq.dtype
        orig_dim = xq.size(-1)
        pad_dim = (2 - orig_dim % 2) % 2

        xq_shape = xq.shape
        xk_shape = xk.shape

        freqs_cis_q = freqs_cis_q[ : xq_shape[-2]]
        freqs_cis_k = freqs_cis_k[ : xk_shape[-2]]
        
        xq_padded, xk_padded = xq, xk
        if pad_dim > 0:
            xq_padded = F.pad(xq.float(), (0, pad_dim))
            xk_padded = F.pad(xk.float(), (0, pad_dim))
        
        xq_ = xq_padded.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk_padded.float().reshape(*xk.shape[:-1], -1, 2)
        
        # --- 直接使用复数计算（假设 freqs_cis 已经是复数）---
        xq_complex = torch.view_as_complex(xq_)  # [..., d//2]
        xk_complex = torch.view_as_complex(xk_)  # [..., d//2]
        
        #seq_len = xq.size(1)

        #if len(freqs_cis.shape) == 2:  # [seq_len, dim//2]
        #    freqs_cis = freqs_cis[:seq_len].unsqueeze(0)  # [1, seq_len, dim//2]
        #elif len(freqs_cis.shape) == 3:  # [batch, seq_len, dim//2]
        #    freqs_cis = freqs_cis[:, :seq_len, :]
        #else:
        #    raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape}")
        #freqs_cis = freqs_cis.unsqueeze(2)

        freqs_cis_q = freqs_cis_q.unsqueeze(0).unsqueeze(0)
        freqs_cis_k = freqs_cis_k.unsqueeze(0).unsqueeze(0)

        xq_rot = xq_complex * freqs_cis_q
        xk_rot = xk_complex * freqs_cis_k

        
        # 转换回实数
        xq_rot_real = torch.view_as_real(xq_rot).flatten(-2)[..., :orig_dim]
        xk_rot_real = torch.view_as_real(xk_rot).flatten(-2)[..., :orig_dim]
        
        return xq_rot_real.to(orig_dtype), xk_rot_real.to(orig_dtype)





def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        torch.unsqueeze(x, dim=2)
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
    
    
def build_norm(norm_type: str, dim: int, eps: float = 1e-6):
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layer_norm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rms_norm":
        return nn.RMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class SlidingCacheAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads=None,
        dropout=0.,
        causal=False,
        max_window_len=None,
        qk_norm=None,
        eps=1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        
        self.to_q = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(self.n_heads * self.head_dim, dim)

        self.dropout = dropout
        self.causal = causal

        self.max_window_len = max_window_len
        self.kv_cache = None
        
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        else:
            self.norm_q = build_norm(norm_type=qk_norm, dim=self.head_dim, eps=eps)
            self.norm_k = build_norm(norm_type=qk_norm, dim=self.head_dim, eps=eps)
    
    def prepare_kv_cache(self, config=None):
        net_device = next(self.parameters()).device
        net_dtype = next(self.parameters()).dtype
        assert config is not None
        self.kv_cache = CustomSlidingCache(
            window_len=self.max_window_len,
            head_dim=self.head_dim, 
            num_attention_heads=self.n_kv_heads,
            device=net_device,
            dtype=net_dtype,
            **config
        )
        
    def clean_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.clean_kv_cache()
            
    def reorder_kv_cache(self, all_chosen_beam_idx):
        if self.kv_cache is not None:
            self.kv_cache.reorder_kv_cache(all_chosen_beam_idx)

    def forward(self, x, cache_kv=False, segment_position=0, freqs_cis=None,alibi=None):

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', d=self.head_dim), (q, k, v))
        attn_mask = None
        if alibi is not None:
            seq_len = q.shape[2]            
            if seq_len == alibi.token_level_alibi.shape[1]:
                attn_mask = alibi.token_level_alibi
            elif seq_len == alibi.face_level_alibi.shape[1]:
                attn_mask = alibi.face_level_alibi
            else:
                attn_mask = alibi.vertex_level_alibi
            attn_mask.requires_grad_(False)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        
        if isinstance(freqs_cis,MultiLevelRoPE):
            seq_len = q.shape[2]
            position_embeddings_q = None
            position_embeddings_k = None
            if seq_len == freqs_cis.face_level_freq.shape[1]:
                position_embeddings_q = freqs_cis.face_level_freq
                
            elif seq_len == freqs_cis.vetrix_level_freq.shape[1]:
                position_embeddings_q = freqs_cis.vetrix_level_freq
            else:
                position_embeddings_q = freqs_cis.token_level_freq
                
            q, k = freqs_cis.apply(q,k,[position_embeddings_q,position_embeddings_q])
        else:
            need_freqs_length = segment_position + q.shape[2]
            assert freqs_cis.shape[0] >= need_freqs_length
            position_embeddings = freqs_cis[segment_position:]
            q, k = apply_rotary_emb(q, k, position_embeddings) 

        if cache_kv:
            pre_k, pre_v = self.kv_cache.recall_memory_kv(k)
            if pre_k.shape[-2] >= self.max_window_len:
                pre_k, pre_v = pre_k[:, :, 1:], pre_v[:, :, 1:]
                
            self.kv_cache.update_cache_kv(k, v)
            k = torch.cat([pre_k, k], dim=-2)
            v = torch.cat([pre_v, v], dim=-2)
        
        causal = self.causal
        attn_implementation = 'sdpa'
        flash_attn_avaliable = k.shape[2] <= self.max_window_len and q.shape[2] == k.shape[2]
        if flash_attn_avaliable:
            sliding_window_size = None
            enable_gqa = self.n_rep > 1
            
            if FLASH_ATTN_3_AVAILABLE:
                attn_implementation = 'fa3'
        else:
            sliding_window_size = self.max_window_len
            enable_gqa = False
            k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

            if not self.training and cache_kv and q.shape[2] == 1 and k.shape[2] <= self.max_window_len:
                sliding_window_size = None
                causal = False

        if alibi is not None:
            attn_implementation = 'sdpa'

        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        out = attention_interface(
            q, k, v, 
            dropout=self.dropout,
            sliding_window_size=sliding_window_size,
            causal=causal,
            enable_gqa=enable_gqa,
            attn_mask=attn_mask
        )  # [B, N, H, D]

        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)
    
    def init_weights(self, init_std: float):
        for linear in (self.to_q, self.to_k, self.to_v):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.to_out.weight, mean=0.0, std=init_std)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        n_heads,
        n_kv_heads=None,
        dropout=0.,
        qk_norm=None,
        eps=1e-6,
        **kwargs
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        
        self.to_q = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(self.n_heads * self.head_dim, dim)

        self.dropout = dropout
        
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        else:
            self.norm_q = build_norm(norm_type=qk_norm, dim=self.head_dim, eps=eps)
            self.norm_k = build_norm(norm_type=qk_norm, dim=self.head_dim, eps=eps)

    def forward(self, x, context,freqs_cis=None):

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', d=self.head_dim), (q, k, v))
        
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        attn_implementation = 'fa3' if FLASH_ATTN_3_AVAILABLE else 'sdpa'
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        out = attention_interface(q, k, v, dropout=self.dropout, enable_gqa=self.n_rep > 1)  # [B, N, H, D]
        
        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)
    
    def init_weights(self, init_std: float):
        for linear in (self.to_q, self.to_k, self.to_v):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.to_out.weight, mean=0.0, std=init_std)


def sdpa_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: float = 0.0,
    sliding_window_size=None,
    causal: Optional[bool] = False,
    enable_gqa=False,
    attn_mask=None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:

    B, Hq, N, D = query.shape
    M = key.shape[2]
    sliding_window_size = None
    
    if sliding_window_size is not None:
        causal = False
        assert M >= N
        bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(_window_size=sliding_window_size)
        
    else:
        if N != M: 
            assert not causal
            causal = False

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout,
        is_causal=causal,
        attn_mask=attn_mask,
        enable_gqa=enable_gqa,
    )  # shape: (B, Hq, N, D)
    
    attn_output = attn_output.to(query.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, N, H, D]
    return attn_output


def flash_attention_3_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: float = 0.0,
    causal: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.Tensor, None]:

    assert dropout == 0
    
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    
    attn_output, _ = flash_attn_func(query, key, value, causal=causal)
    return attn_output


ALL_ATTENTION_FUNCTIONS: Dict[str, Callable] = {
    'sdpa': sdpa_attention_forward,
    'fa3': flash_attention_3_forward,
}
