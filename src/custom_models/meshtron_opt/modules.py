import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from einops import rearrange, reduce, repeat

from src.custom_models.meshtron_opt.attention import SlidingCacheAttention, CrossAttention, build_norm


class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, 'b (n s) d -> b n d', 'mean', s=self.shorten_factor)


class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'b n d -> b (n s) d', s=self.shorten_factor)


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s=self.shorten_factor)
    
class Conv1DDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim,3,3)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = x.permute(0,2,1)
        return self.proj(x).permute(0,2,1)  # b n (s d) -> b (n s) d


class Conv1DUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.ConvTranspose1d(dim, dim,3,3)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.proj(x)
        x = x.permute(0,2,1) # 'b n (s d) -> b (n s) d'
        return x
    
class SwiFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult, bias=False)
        self.w2 = nn.Linear(dim * mult, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * mult, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)



class DiscreteValueEmbeding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.value_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.value_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, discrete_value):
        value_proj = self.value_proj(discrete_value)
        hidden_dtype = self.value_embedder.linear_1.weight.dtype
        value_emb = self.value_embedder(value_proj.to(dtype=hidden_dtype))  # (N, D)
        return value_emb




class TransformerFlowBlock(nn.Module):
    def __init__(
        self, 
        layer_id: int,
        dim,
        context_dim=None,
        n_heads=8,
        n_kv_heads=None,
        ffn_multiple=4,
        qk_norm=None,
        max_window_len=None,
        eps=1e-6,
        depth_init=False,
    ):
        """
        Args:
            context_dim: 为 None 时，该 block 不引入 cross_attention，否则将在 self_attention 后增加一个 cross_attention 模块。
        """
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        
        # self attention 模块
        self.norm1 = build_norm(norm_type='rms_norm', dim=dim, eps=eps)
        self.attn1 = SlidingCacheAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, causal=False, 
                                           max_window_len=max_window_len, qk_norm=qk_norm, eps=eps)

        # cross attention 模块
        if context_dim is not None:
            self.norm2 = build_norm(norm_type='rms_norm', dim=dim, eps=eps)
            self.attn2 = CrossAttention(dim=dim, context_dim=context_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, 
                                        qk_norm=qk_norm, eps=eps)
        else:
            self.norm2, self.attn2 = None, None
        
        self.norm3 = build_norm(norm_type='rms_norm', dim=dim, eps=eps)
        self.attn3 = CrossAttention(dim=dim, context_dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, 
                                        qk_norm=qk_norm, eps=eps) # for token-vertrix-face
            
        # ffn 模块
        self.ffn_norm = build_norm(norm_type='rms_norm', dim=dim, eps=eps)
        self.feed_forward = SwiFeedForward(dim=dim, mult=ffn_multiple)

        if depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            # reference from DeepSeekV3: All learnable parameters are randomly initialized with a standard deviation of 0.006.
            self.weight_init_std = 0.006

    def forward(self, x, context=None, context_level=None, cache_kv=False, segment_position=0, freqs_cis=None,alibi=None):
        # self attention
        norm_x = self.norm1(x)
        attn_output = self.attn1(norm_x, cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi)
        x = x + attn_output

        # cross attention for token-vertrix-face
        if self.attn3 is not None and context_level is not None:
            norm_x = self.norm3(x)
            attn_output = self.attn3(norm_x, context=context_level,freqs_cis=freqs_cis)
            x = x + attn_output
        # cross attention
        if self.attn2 is not None:
            norm_x = self.norm2(x)
            attn_output = self.attn2(norm_x, context=context)
            x = x + attn_output
            
        # feed forward
        norm_x = self.ffn_norm(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        return x

    def init_weights(self):
        for norm in (self.norm1, self.norm2, self.ffn_norm):
            if norm is None:
                continue
            norm.reset_parameters()
        
        self.attn1.init_weights(self.weight_init_std)
        if self.attn2 is not None:
            self.attn2.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)
        
        
class TransformerFlow(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        n_heads=8,
        n_kv_heads=None,
        depth=12,
        condition_interval=4,
        ffn_multiple=4,
        qk_norm=None,
        max_window_len=None,
        depth_init=False,
        grad_checkpoint_interval=1,
    ):
        super().__init__()
        
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(depth):
            # 确认当前 block 是否需要 cross attention
            if context_dim is not None and condition_interval and layer_id % condition_interval == 0:
                cross_attn_dim = context_dim
            else:
                cross_attn_dim = None
            # 初始化 block
            layer_name = str(layer_id)
            self.layers[layer_name] = TransformerFlowBlock(
                layer_id=layer_id,
                dim=dim,
                context_dim=cross_attn_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ffn_multiple=ffn_multiple,
                qk_norm=qk_norm,
                max_window_len=max_window_len,
                depth_init=depth_init,
            )
        # transformer 的输出可能会被用于 downsample, upsample, to_logits 等，最好进行一次 norm
        self.norm = build_norm(norm_type='rms_norm', dim=dim, eps=1e-6)

        self.grad_checkpoint_interval = grad_checkpoint_interval
        self.gradient_checkpointing = False
        self.init_weights()
        
    def forward(self, x, context=None, context_level=None, cache_kv=False, segment_position=0, freqs_cis=None,alibi=None):
        for layer_name, layer in self.layers.items():
            if self.gradient_checkpointing and self.training and int(layer_name) % self.grad_checkpoint_interval == 0:
                def create_custom_forward(module, **kwargs):
                    # custom_forward 的入参只能是 *args
                    def custom_forward(*inputs):
                        return module(*inputs, **kwargs)
                    return custom_forward
                
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, context=context,
                                                 context_level=context_level,
                                                 cache_kv=cache_kv,
                                                 segment_position=segment_position,
                                                 freqs_cis=freqs_cis,
                                                 alibi=alibi),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, context=context,context_level=context_level,cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi)
        return self.norm(x)

    def init_weights(self):
        for layer in self.layers.values():
            layer.init_weights()

    def prepare_kv_cache(self, config=None):
        for layer in self.layers.values():
            layer.attn1.prepare_kv_cache(config)

    def clean_kv_cache(self):
        for layer in self.layers.values():
            layer.attn1.clean_kv_cache()

    def reorder_kv_cache(self, all_chosen_beam_idx):
        for layer in self.layers.values():
            layer.attn1.reorder_kv_cache(all_chosen_beam_idx)

