import random
import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.utils.common import CROSS_ENTROPY_IGNORE_IDX


def discretize(
    t: Tensor,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)    # cube normalize
    t *= num_discrete
    t -= 0.5
    return t.round().long().clamp(min = 0, max = num_discrete - 1)


def undiscretize(
    t: Tensor,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = t.float()
    t += 0.5
    t /= num_discrete       # cube normalize
    return t * (hi - lo) + lo


@dataclass
class SplitResult:

    input_ids: torch.LongTensor = None
    labels: torch.LongTensor = None
    

class MeshTokenizer(ConfigMixin):
    config_name = 'tokenizer_config.json'
    
    @register_to_config
    def __init__(self, n_discrete_size, bos_len, eos_len, n_sliding_triangles, split_method):

        super().__init__()
        self.pad_id = -1
        self.num_discrete_coors = n_discrete_size  # default: 800
        self.codebook_size = n_discrete_size       # default: 128
        self.coor_continuous_range = (-0.5, 0.5) 
        self.bos_len = bos_len  
        self.eos_len = eos_len 

        self.face_place_holder = None  
        self.end_face_holder = None
        
        self.vocab_size = self.codebook_size + 4
        self.bos_token_id = self.codebook_size
        self.eos_token_id = self.codebook_size + 1
        self.pad_token_id = self.codebook_size + 2
        self.mask_token_id = self.codebook_size + 3
        
        assert split_method in ['random', 'uniform']
        self.split_method = split_method

    def tokenize(self, data_dict: dict) -> dict:

        ### 3D mesh face parsing
        vertices = data_dict['vertices']    # batch x nv x 3
        faces = data_dict['faces']          # batch x nf x 3
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')   # batch x nf
        
        assert vertices.dtype in [torch.float16, torch.float32, torch.float64], f"{vertices.dtype} is not enough to discretize"
        
        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        
        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)
        face_coords = vertices.gather(-2, faces_vertices.long())
        
        discrete_face_coords = face_coords.long()
        assert (discrete_face_coords >= 0).all() and ((discrete_face_coords - face_coords).abs() < 1e-5).all()
        
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            self.pad_id
        )
        
        input_ids = discrete_padded_coords.reshape(batch, -1)
        attention_mask = (input_ids != self.pad_id).float()

        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        self.start_place_holder = torch.cat([place_holder] * self.bos_len, dim=1)
        self.end_face_holder = torch.cat([place_holder] * self.eos_len, dim=1)
        input_ids = torch.cat((self.start_place_holder * self.pad_id, input_ids, self.end_face_holder * self.pad_id), dim=1)
        attention_mask = torch.cat((self.end_face_holder, self.start_place_holder, attention_mask), dim=1)
        
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        
        input_ids[input_ids == self.pad_id] = self.pad_token_id   
        input_ids[:, :self.bos_len] = self.bos_token_id                                
        eos_pos_id = attention_mask.sum(1, keepdim=True) - self.eos_len   

        for i in range(self.eos_len):
            input_ids = torch.scatter(                                         
                input_ids, 
                1, 
                (eos_pos_id + i).long(), 
                torch.ones_like(input_ids) * self.eos_token_id
            )

        b, n = input_ids.shape
        if (n % 9) != 0:
            reminder = n % 9
            input_ids = input_ids[:, :-reminder]
            attention_mask = attention_mask[:, :-reminder]
        
        assert input_ids.shape[1] % 9 == 0 and attention_mask.shape==input_ids.shape
        
        return input_ids.long(), attention_mask.long()
    
    def detokenize(self, input_ids: Tensor) -> dict:

        input_ids = input_ids.reshape(input_ids.shape[0], -1)

        face_mask = reduce(
            input_ids != self.pad_token_id, 'b (nf c) -> b nf', 'all', c = 9
        )
        
        pred_face_coords = input_ids.reshape(input_ids.shape[0], -1, 9)
        pred_face_coords = rearrange(
            pred_face_coords, '... (v c) -> ... v c', v = 3
        )
        
        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete=self.num_discrete_coors,
            continuous_range=self.coor_continuous_range
        )
        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            float('nan')
        )
        output_dict = {}
        output_dict['recon_faces'] = continuous_coors
        return output_dict
    
    def preprocess(self, data_dict, remove_tail=True, split=True, need_sample_num=1,
                   input_ids=None, attention_mask=None):

        if input_ids is None or attention_mask is None:
            input_ids, attention_mask = self.tokenize(data_dict)
        if remove_tail:
            input_ids, attention_mask = self.remove_tail_token(input_ids, attention_mask)
        if not split:
            return input_ids, attention_mask
        
        max_window_len = self.config.n_sliding_triangles * 9
        if input_ids.shape[1] <= max_window_len:
            target = self.construct_target(input_ids, attention_mask)
            split_result = SplitResult(input_ids=input_ids, labels=target)
            return split_result
        
        if self.split_method == 'random':
            all_split_results = self.random_split(input_ids, attention_mask)
        elif self.split_method == 'uniform':
            all_split_results = self.uniform_split(input_ids, attention_mask)
        else:
            raise ValueError(f"unknown split_method: {self.split_method}.")
        
        if need_sample_num is None:
            return all_split_results
        else:
            need_sample_num = min(need_sample_num, len(all_split_results))
            assert need_sample_num == 1
            sampled_results = np.random.choice(all_split_results, size=need_sample_num, replace=False)[0]
            return sampled_results
    
    def random_split(self, input_ids, attention_mask):

        bsz, _ = input_ids.shape
        front_triangle_num = random.randint(0, self.config.n_sliding_triangles - 64)
        front_token_num = front_triangle_num * 9
        front_ids = torch.ones((bsz, front_token_num), dtype=input_ids.dtype, device=input_ids.device) * -1
        front_mask = torch.zeros((bsz, front_token_num), dtype=attention_mask.dtype, device=attention_mask.device)
        
        concated_ids = torch.concat([front_ids, input_ids], dim=1)
        concated_mask = torch.concat([front_mask, attention_mask], dim=1)
        all_split_results = self.uniform_split(input_ids=concated_ids, attention_mask=concated_mask)

        first_split_result = all_split_results[0]
        front_pos = first_split_result.input_ids == -1
        assert (front_token_num == 0 or front_pos[:, :front_token_num].all()) and not front_pos[:, front_token_num:].any()
        
        all_split_results[0] = SplitResult(
            input_ids=first_split_result.input_ids[:, front_token_num:],
            labels=first_split_result.labels[:, front_token_num:],
        )
        return all_split_results
    
    def uniform_split(self, input_ids, attention_mask):

        max_window_len = self.config.n_sliding_triangles * 9
        split_ids, split_last_target_id = self.split_sequence(input_ids, max_window_len)
        split_mask, split_last_target_mask = self.split_sequence(attention_mask, max_window_len)
        
        all_split_results = []
        for i in range(len(split_ids)):
            target = self.construct_target(
                input_ids=split_ids[i],
                attention_mask=split_mask[i],
                last_target_id=split_last_target_id[i],
                last_target_mask=split_last_target_mask[i],
            )
            split_result = SplitResult(input_ids=split_ids[i], labels=target)
            all_split_results.append(split_result)
        
        return all_split_results
    
    @staticmethod
    def split_sequence(sequence, window_len):

        normal_seqs = sequence.split(window_len, dim=-1)
        split_res, last_target = [], []
        for i in range(len(normal_seqs)):
            sub_seq = normal_seqs[i]
            split_res.append(sub_seq)
            if i == len(normal_seqs) - 1:
                last_target.append(None)
            else:
                next_seq = normal_seqs[i + 1]
                last_target.append(next_seq[:, :1])
        return split_res, last_target
    
    @staticmethod
    def remove_tail_token(input_ids, attention_mask):
        valid_length = int(torch.sum(attention_mask, dim=-1).max())
        input_ids = input_ids[:, :valid_length]
        attention_mask = attention_mask[:, :valid_length]
        return input_ids, attention_mask
    
    @staticmethod
    def construct_target(input_ids, attention_mask, last_target_id=None, last_target_mask=None):
        part_target = torch.where(attention_mask[:, 1:] > 0, input=input_ids[:, 1:], other=CROSS_ENTROPY_IGNORE_IDX)
        if last_target_id is None:
            last_target = torch.ones_like(input_ids[:, 0:1]) * CROSS_ENTROPY_IGNORE_IDX
        else:
            last_target = torch.where(last_target_mask > 0, input=last_target_id, other=CROSS_ENTROPY_IGNORE_IDX)
        target = torch.concat([part_target, last_target], dim=1)
        return target.long()

    def save_pretrained(self, save_directory):
        self.save_config(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        output_loading_info = kwargs.pop("output_loading_info", False)
        config, unused_kwargs = cls.load_config(
            pretrained_model_path,
            return_unused_kwargs=True,
            **kwargs,
        )
        tokenizer = cls.from_config(config, **unused_kwargs)
        
        if output_loading_info:
            return tokenizer, f"final config: {tokenizer.config}"
        return tokenizer
