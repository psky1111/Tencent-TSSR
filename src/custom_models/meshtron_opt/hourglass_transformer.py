import torch.nn.functional as F
from torch import nn

from src.custom_models.meshtron_opt.modules import (
    TransformerFlow,
    NaiveDownsample, 
    NaiveUpsample,
    LinearDownsample, 
    LinearUpsample, 
    Conv1DDownsample,
    Conv1DUpsample,
)
from src.custom_models.meshtron_opt.cache_utils import RoutingCache


class HourglassCasualSlidingTransformerFlow(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        *,
        n_heads=8,
        n_kv_heads=None,
        depth,
        shorten_factor=3,
        updown_sample_type='linear',
        max_window_len=None,
        ffn_multiple=4,
        qk_norm=None,
        condition_interval=4,
    ):
        super().__init__()
        
        pre_layers_depth, valley_depth, post_layers_depth = depth  # 分别对应D端blocks，中间，U端blocks
        self.shorten_factor = shorten_factor
        
        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample = LinearUpsample(dim, shorten_factor)
        elif updown_sample_type == "conv":
            self.downsample = Conv1DDownsample(dim, shorten_factor)
            self.upsample = Conv1DUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')
        
        transformer_kwargs = dict(
            dim=dim,
            context_dim=context_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_multiple=ffn_multiple,
            qk_norm=qk_norm,
            condition_interval=condition_interval,
        )

        if isinstance(valley_depth, int):
            self.valley_transformer = TransformerFlow(
                depth=valley_depth, 
                max_window_len=max_window_len // shorten_factor, 
                **transformer_kwargs
            )
        else:
            self.valley_transformer = HourglassCasualSlidingTransformerFlow(
                shorten_factor=shorten_factor,
                depth=valley_depth,
                updown_sample_type=updown_sample_type,
                max_window_len=max_window_len // shorten_factor,
                **transformer_kwargs
            )
        
        self.pre_transformer = TransformerFlow(depth=pre_layers_depth, max_window_len=max_window_len, **transformer_kwargs)
        self.post_transformer = TransformerFlow(depth=post_layers_depth, max_window_len=max_window_len, **transformer_kwargs)

        self.routing_cache = RoutingCache(shorten_factor)
    
    def clean_routing_cache(self):
        pass
    
    def prepare_kv_cache(self, config):
        pass

    def clean_kv_cache(self,):
        pass
        
    def reorder_kv_cache(self, all_chosen_beam_idx):
        pass
        
    def reorder_routing_cache(self, all_chosen_beam_idx):
        pass

    def casual_shift(self, x, left_shift, right_shift):
        return x
    
    def forward(
        self, 
        x, 
        context=None, 
        context_level=None,
        cache_kv=False,
        cache_position=None,
        is_sampling=False,
        segment_position=0,
        freqs_cis=None,
        alibi=None,
        **kwargs
    ):

        if is_sampling:
            assert cache_position is not None
        
        s, b, n = self.shorten_factor, *x.shape[:2]
        x = self.pre_transformer(x, context,context_level=context_level, cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi) 
        if is_sampling:
            x = self.routing_cache.update_prev(x)

        x_residual = x.clone()
        if x.shape[1] % s == 0:
            x = self.downsample(x)

            if isinstance(self.valley_transformer, TransformerFlow):
                x = self.valley_transformer(
                    x, 
                    context=context, 
                    context_level=x_residual,
                    cache_kv=cache_kv,
                    segment_position=segment_position // s,
                    freqs_cis=freqs_cis,
                    alibi=alibi,
                )
            else:
                x = self.valley_transformer(
                    x, 
                    context=context, 
                    context_level=x_residual,
                    cache_kv=cache_kv,
                    cache_position=cache_position // s if cache_position is not None else cache_position,
                    segment_position=segment_position // s,
                    is_sampling=is_sampling,
                    freqs_cis=freqs_cis,
                    alibi=alibi,
                ) 

            up_x = self.upsample(x)
            if is_sampling:
                self.routing_cache.memory_route(x[:, -s:].clone())

            up_x = up_x + x_residual
            if is_sampling and cache_kv and n == 1:
                up_x = up_x[:, -1:]
        else:
            assert is_sampling
            memory = self.routing_cache.recall_route(cache_position % s, cache_kv, x_residual)
            up_x = memory + x_residual

        x = self.post_transformer(up_x, context, context_level=x, cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi)
        return x





class HourglassCasualSlidingSelfCorrectionTransformerFlow(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        *,
        n_heads=8,
        n_kv_heads=None,
        depth,
        shorten_factor=3,
        updown_sample_type='linear',
        max_window_len=None,
        ffn_multiple=4,
        qk_norm=None,
        condition_interval=4,
    ):
        super().__init__()
        
        pre_layers_depth, valley_depth, post_layers_depth = depth  
        self.shorten_factor = shorten_factor
        
        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample = LinearUpsample(dim, shorten_factor)
        elif updown_sample_type == "conv":
            self.downsample = Conv1DDownsample(dim, shorten_factor)
            self.upsample = Conv1DUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        transformer_kwargs = dict(
            dim=dim,
            context_dim=context_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_multiple=ffn_multiple,
            qk_norm=qk_norm,
            condition_interval=condition_interval,
        )

        if isinstance(valley_depth, int):
            self.valley_transformer = TransformerFlow(
                depth=valley_depth, 
                max_window_len=max_window_len // shorten_factor, 
                **transformer_kwargs
            )
        else:
            self.valley_transformer = HourglassCasualSlidingTransformerFlow(
                shorten_factor=shorten_factor,
                depth=valley_depth,
                updown_sample_type=updown_sample_type,
                max_window_len=max_window_len // shorten_factor,
                **transformer_kwargs
            )
        
        self.pre_transformer = TransformerFlow(depth=pre_layers_depth, max_window_len=max_window_len, **transformer_kwargs)
        self.post_transformer = TransformerFlow(depth=post_layers_depth, max_window_len=max_window_len, **transformer_kwargs)

        self.routing_cache = RoutingCache(shorten_factor)
    
    def clean_routing_cache(self):
        self.routing_cache.clean_routing_cache()
        if isinstance(self.valley_transformer, TransformerFlow):
            pass
        else:
            self.valley_transformer.clean_routing_cache()
    
    def prepare_kv_cache(self, config):
        self.pre_transformer.prepare_kv_cache(config)
        self.post_transformer.prepare_kv_cache(config)
        self.valley_transformer.prepare_kv_cache(config)

    def clean_kv_cache(self,):
        self.pre_transformer.clean_kv_cache()
        self.post_transformer.clean_kv_cache()
        self.valley_transformer.clean_kv_cache()
        
    def reorder_kv_cache(self, all_chosen_beam_idx):
        self.pre_transformer.reorder_kv_cache(all_chosen_beam_idx)
        self.post_transformer.reorder_kv_cache(all_chosen_beam_idx)
        self.valley_transformer.reorder_kv_cache(all_chosen_beam_idx)
        
    def reorder_routing_cache(self, all_chosen_beam_idx):
        self.routing_cache.reorder_routing_cache(all_chosen_beam_idx)
        if isinstance(self.valley_transformer, TransformerFlow):
            pass
        else:
            self.valley_transformer.reorder_routing_cache(all_chosen_beam_idx)

    def casual_shift(self, x, left_shift, right_shift):
        shifted_x = F.pad(x, (0, 0, left_shift, -right_shift), value=0.)
        return shifted_x
    
    def forward(
        self, 
        x, 
        context=None, 
        context_level=None,
        cache_kv=False,
        cache_position=None,
        is_sampling=False,
        segment_position=0,
        freqs_cis=None,
        alibi=None,
        **kwargs
    ):

        if is_sampling:
            assert cache_position is not None
        
        s, b, n = self.shorten_factor, *x.shape[:2]
        x = self.pre_transformer(x, context,context_level=context_level, cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi) 
        if is_sampling:
            x = self.routing_cache.update_prev(x)

        x_residual = x.clone()
        if x.shape[1] % s == 0:
            x = self.downsample(x)

            if isinstance(self.valley_transformer, TransformerFlow):
                x = self.valley_transformer(
                    x, 
                    context=context, 
                    context_level=context_level,
                    cache_kv=cache_kv,
                    segment_position=segment_position // s,
                    freqs_cis=freqs_cis,
                    alibi=alibi,
                )
            else:
                x = self.valley_transformer(
                    x, 
                    context=context, 
                    context_level=context_level,
                    cache_kv=cache_kv,
                    cache_position=cache_position // s if cache_position is not None else cache_position,
                    segment_position=segment_position // s,
                    is_sampling=is_sampling,
                    freqs_cis=freqs_cis,
                    alibi=alibi,
                ) 

            up_x = self.upsample(x)
            if is_sampling:
                self.routing_cache.memory_route(x[:, -s:].clone())

            up_x = up_x + x_residual
            if is_sampling and cache_kv and n == 1:
                up_x = up_x[:, -1:]
        else:
            assert is_sampling
            memory = self.routing_cache.recall_route(cache_position % s, cache_kv, x_residual)
            up_x = memory + x_residual

        x = self.post_transformer(up_x, context, context_level=context_level, cache_kv=cache_kv, segment_position=segment_position, freqs_cis=freqs_cis,alibi=alibi)
        return x
