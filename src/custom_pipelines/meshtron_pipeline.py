import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from typing import Dict

from src.custom_models.meshtron_opt.flow_mask_meshtron import MeshtronNet
from src.custom_models.meshtron_opt.tokenizer import MeshTokenizer, undiscretize
from src.utils.common import init_logger
from src.utils.mesh import z_up_coord_to_y_up



logger = init_logger()


class MeshtronPipeline:
    def __init__(self, tokenizer, meshtron_net, point_encoder, device='cuda', dtype=torch.bfloat16):
        self.tokenizer: MeshTokenizer = tokenizer
        self.meshtron_net: MeshtronNet = meshtron_net
        self.point_encoder = point_encoder
        self.device = device
        self.dtype = dtype
        self.burn_in_triangles = None
        self.bad_version = None
    
    @torch.no_grad()
    def __call__(
        self, 
        point_cond,
        face_count_id,
        n_max_triangles,
        filter_thres=0.9,
        temperature=1.0,
        inference_num=200,
        cfg=1.5,
        t=0.01,
        gt_vertices=None,
        batch_data=None,
        generator=None,
        verify_consistency=False,
        use_kv_cache=True,
        make_nll_curve=False,
        gen_mesh=True,
        calc_ppl=False,
        burn_in_triangles=None,
        degeneration_burn_in_triangles=None,
        beam_num=None,
        constrain_sample=False,
        enable_bad_model=False,
        **kwargs
    ):
        """
        Args:
            gen_mesh: 是否生成 mesh，在 debug 中有时只想观察 ppl 结果，可以将此参数设为 False。
            calc_ppl: 是否计算 ppl, 在实际使用中有时只想快速生成 mesh 结果，可以将此参数设为 False。
            burn_in_triangles: 若不为 None，则每推理一定数量三角面片后，清空 kv cache，并使用最新生成的 burn_in_triangles 个面片进行 prefill。
            degeneration_burn_in_triangles: 若不为 None，则检测到退化情况时，去掉退化面，并进行一定数量的 burn-in，与 burn_in_triangles 参数
                不能同时指定。
        """
        # 每一次 call 都需要刷新 self.burn_in_triangles，消除之前的推理设置
        #self.burn_in_triangles = burn_in_triangles
        #self.degeneration_burn_in_triangles = degeneration_burn_in_triangles
        # burn_in_triangles 为生成一定数量面数后激活，degeneration_burn_in_triangles 为检测到退化结果时激活，暂时设定为不能同时生效
        #assert not (self.burn_in_triangles is not None and self.degeneration_burn_in_triangles is not None)
        
        # 控制 point_encoder 编码中 fps 操作的随机性
        new_point_cond = []
        for idx in range(point_cond.shape[0]):
            # randperm 似乎无法接受 cuda generator
            cpu_generator = torch.Generator().manual_seed(generator.initial_seed()) if generator is not None else torch.Generator()
            perm = torch.randperm(point_cond.shape[1], generator=cpu_generator)
            new_point_cond.append(point_cond[idx][perm])
        point_cond = torch.stack(new_point_cond)
        context = self.point_encoder(point_cond, random_fps=False)
        
        result_dict, metric_dict = {}, {}
        gt_vertices = torch.flip(gt_vertices, dims=(-1,))  # vertices 数值为量化后的值，范围是 [0, reso - 1], shape: (b, n_face, 3, 3)
        gt_vertices = undiscretize(
            gt_vertices,
            num_discrete=self.tokenizer.num_discrete_coors,
            continuous_range=self.tokenizer.coor_continuous_range
        )  # 将 gt 顶点缩放到 (-0.5, 0.5)
        input_ids, attention_mask = self.tokenizer.preprocess(batch_data, need_sample_num=1,split=False)

        if enable_bad_model:
            if not self.bad_version:
                self.bad_version = MeshtronNet.from_pretrained(kwargs.get("bad_model_path", None)).to(self.device)
        
        # 生成 mesh
        if gen_mesh:
            max_seq_length = self.tokenizer.bos_len + n_max_triangles * 9 + self.tokenizer.bos_len
            out = self._generate_discrete_flow(max_seq_length,context,t,inference_num,input_ids=input_ids,face_count_id=face_count_id,cfg=cfg,enable_bad_model=enable_bad_model,bad_model=self.bad_version)

            if isinstance(out, list):
                for i in range(len(out)):
                    out[i] = self.postprocess_output_ids(out[i])
                generated_meshes = out[-1]

            else:
                generated_meshes = self.postprocess_output_ids(out)

            

            if isinstance(out, list):
                for i in range(len(out)):
                    result_dict[f"generated_mesh_{i}"] = {'data': z_up_coord_to_y_up(out[i].cpu().to(torch.float16)), 'visualize': 'mesh'}
            else:
                result_dict.update(
                    generated_mesh={'data': z_up_coord_to_y_up(generated_meshes.cpu().to(torch.float16)), 'visualize': 'mesh'},
                )


        # 写入 gt 信息
        result_dict.update(
            point_cond={'data': z_up_coord_to_y_up(point_cond[:, :, :3].cpu().to(torch.float16)), 'visualize': 'point_cloud'},
            gt_mesh={'data': z_up_coord_to_y_up(gt_vertices.cpu()), 'visualize': 'mesh'},
        )
        return result_dict, metric_dict
    
    def postprocess_output_ids(self, out):
        # mask out everything after the eos tokens
        is_eos_tokens = (out == self.tokenizer.eos_token_id)
        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
        mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
        out = out.masked_fill(mask, self.tokenizer.eos_token_id)
        # discard <bos> and <eos> tokens to pad tokens
        #output_ids = out[:, self.tokenizer.bos_len: -1]
        output_ids = out
        output_ids[output_ids == self.tokenizer.bos_token_id] = self.tokenizer.pad_token_id
        output_ids[output_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id

        _, n = output_ids.shape
        remainder = n % 9
        if remainder != 0:
            output_ids = output_ids[:, :-remainder]

        decoder_output = self.tokenizer.detokenize(input_ids=output_ids)
        generated_meshes = decoder_output["recon_faces"]  # shape: (nsample, nf, 3, 3)
        
        # 生成的 mesh 与 gt mesh 均进行了点坐标 (x, y, z) -> (z, y, x) 的转换，现在结果处理阶段需要转换回来
        generated_meshes = torch.flip(generated_meshes, dims=(-1,))
        return generated_meshes
    




    @torch.no_grad()
    def _generate_discrete_flow(self, max_seq_length, context, t,inference_num=1000,input_ids=None,face_count_id=None,cfg=1.0,enable_bad_model=False,**kwargs):
        if t is not None:
            mesh = self.meshtron_net.sample_with_timesteps(context,t,inference_num,1,max_seq_length,input_ids=input_ids,face_count_id=face_count_id,cfg_scale=cfg,enable_bad_model=enable_bad_model,**kwargs)
        else:
            mesh = self.meshtron_net.sample(context,inference_num,1,max_seq_length,face_count_id=face_count_id,cfg_scale=cfg,enable_bad_model=enable_bad_model,**kwargs)
        return mesh
        
