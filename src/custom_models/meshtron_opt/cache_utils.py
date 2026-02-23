import gc
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class RoutingCache(nn.Module):
    def __init__(self, s):
        super().__init__()
        assert s in [3, 9]
        self.shorten_factor = s
        self.routing_cache = None
        self.previous_cache = []
    
    def clean_routing_cache(self):
        self.routing_cache = None
        self.previous_cache = [] 

    def update_prev(self, x):

        if x.shape[-2] == 1: 
            self.previous_cache.append(x)
            if len(self.previous_cache) == self.shorten_factor:
                x = torch.cat(self.previous_cache, dim=-2)
                self.previous_cache = []
        else:
            pass

        return x

    def memory_route(self, x):
        self.routing_cache = x

    def recall_route(self, cache_position, cache_kv, x_residual):
        if self.routing_cache is None: 
            return 0

        cache_position = (cache_position + 1) % self.shorten_factor
        assert cache_position != 0

        if cache_kv:
            assert self.routing_cache.shape[1] == self.shorten_factor
            return self.routing_cache[:, [cache_position], :]
        else:
            trunc_len = x_residual.shape[-2]
            shifted_routing_cache = F.pad(self.routing_cache, (0, 0, self.shorten_factor - 1, 0), value = 0.)
            shifted_routing_cache = shifted_routing_cache[:, :trunc_len]
            return shifted_routing_cache
        
    def reorder_routing_cache(self, all_chosen_beam_idx):
        if self.routing_cache is not None:
            self.routing_cache = torch.index_select(self.routing_cache, dim=0, index=all_chosen_beam_idx)
        
        if len(self.previous_cache) > 0:
            for i in range(len(self.previous_cache)):
                self.previous_cache[i] = torch.index_select(self.previous_cache[i], dim=0, index=all_chosen_beam_idx)


class CustomSlidingCache:
    def __init__(self, window_len, head_dim, num_attention_heads, batch_size, device=None, dtype=torch.bfloat16):
        self.max_cache_len = window_len
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.max_batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        cache_shape = (self.max_batch_size, self.num_attention_heads, self.max_cache_len, self.head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.cache_tail_index = 0
        
    def update_cache_kv(self, key_states: torch.Tensor, value_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        if self.key_cache.device != key_states.device or self.key_cache.dtype != key_states.dtype:
            self.key_cache = self.key_cache.to(device=key_states.device, dtype=key_states.dtype)
            self.value_cache = self.value_cache.to(device=value_states.device, dtype=value_states.dtype)
        
        old_k_cache, old_v_cache = self.recall_memory_kv(key_states)
        new_full_k = torch.concat([old_k_cache, key_states], dim=2)  # shape: (b, h, n, d)
        new_full_v = torch.concat([old_v_cache, value_states], dim=2)
        
        bz, _, full_len, _ = new_full_k.shape
        if full_len >= self.max_cache_len:
            k_out = new_full_k[:bz, :, -self.max_cache_len:]
            v_out = new_full_v[:bz, :, -self.max_cache_len:]
            self.cache_tail_index = self.max_cache_len
        else:
            k_out, v_out = new_full_k, new_full_v
            self.cache_tail_index = full_len
        self.key_cache = k_out
        self.value_cache = v_out
        return k_out, v_out
    
    def recall_memory_kv(self, k):
        cache_len = self.cache_tail_index  
        bz = k.shape[0]
        mem_key = self.key_cache[:bz, :, :cache_len, :].to(device=k.device, dtype=k.dtype)
        mem_value = self.value_cache[:bz, :, :cache_len, :].to(device=k.device, dtype=k.dtype)
        return mem_key, mem_value
    
    def clean_kv_cache(self):
        cache_shape = (self.max_batch_size, self.num_attention_heads, 1, self.head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.cache_tail_index = 0
        
    def reorder_kv_cache(self, all_chosen_beam_idx):
        self.key_cache = torch.index_select(self.key_cache, dim=0, index=all_chosen_beam_idx)
        self.value_cache = torch.index_select(self.value_cache, dim=0, index=all_chosen_beam_idx)
