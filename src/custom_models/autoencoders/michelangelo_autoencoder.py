import math
import random
import torch
import torch.nn as nn
import numpy as np

from einops import repeat, rearrange
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch_cluster import fps
from typing import Any, Dict, Optional, Union, Tuple, List
from jaxtyping import Float, Bool

from src.custom_models.autoencoders.attention import ResidualCrossAttentionBlock, ResidualAttentionBlock
from src.utils.embeddings import get_embedder, FourierEmbedder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 11.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(self,
                 use_downsample: bool,
                 num_latents: int,
                 embedder_out_dim: int,
                 point_feats: int,
                 embed_point_feats: bool,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 use_flash: bool = False,
                 use_multi_reso: bool = False,
                 resolutions: list = [],
                 sampling_prob: list = [],
                 qk_norm=False):

        super().__init__()

        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats
        self.use_multi_reso = use_multi_reso
        self.resolutions = resolutions
        self.sampling_prob = sampling_prob

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        if self.embed_point_feats:
            self.input_proj = nn.Linear(embedder_out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(embedder_out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            qk_norm=qk_norm,
        )

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            qk_norm=qk_norm,
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None
        
        self.gradient_checkpointing = False

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None, embedder=None, random_fps=True):
        """
        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """
        bs, N, D = pc.shape
        
        data = embedder(pc)
        if feats is not None:
            if self.embed_point_feats:
                feats = embedder(feats)
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        if self.use_multi_reso:
            # number = 8192
            resolution = random.choice(self.resolutions, size=1, p=self.sampling_prob)[0]
            if resolution != N:
                flattened = pc.view(bs * N, D)  # bs*N, 64.      103,4096,3 -> 421888,3
                batch = torch.arange(bs).to(pc.device)  # 103
                batch = torch.repeat_interleave(batch, N)  # bs*N. 421888
                pos = flattened
                ratio = 1.0 * resolution / N  # 0.0625
                idx = fps(pos, batch, ratio=ratio)  # 26368
                pc = pc.view(bs * N, -1)[idx].view(bs, -1, D)
                bs, N, D = feats.shape
                flattened1 = feats.view(bs * N, D)
                feats = flattened1.view(bs * N, -1)[idx].view(bs, -1, D)
                bs, N, D = pc.shape

        if self.use_downsample:
            ###### fps
            flattened = pc.view(bs * N, D)  # bs*N, 64

            batch = torch.arange(bs).to(pc.device)
            batch = torch.repeat_interleave(batch, N)  # bs*N

            pos = flattened

            ratio = 1.0 * self.num_latents / N
            if pos.dtype == torch.bfloat16:
                pos = pos.to(dtype=torch.float16)
            idx = fps(pos, batch, ratio=ratio, random_start=random_fps)

            query = data.view(bs * N, -1)[idx].view(bs, -1, data.shape[-1])
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            ckpt_kwargs = {"use_reentrant": False}
            latents = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.cross_attn),
                query,
                data,
                **ckpt_kwargs,
            )
        else:
            latents = self.cross_attn(query, data)

        # self_attn 为 Perceiver 类实例，内部已实现 checkpoint
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents
    

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False,
        qk_norm=False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    use_flash=use_flash,
                    qk_norm=qk_norm,
                )
                for _ in range(layers)
            ]
        )
        
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **ckpt_kwargs,
                )
            else:
                x = block(x)
        return x


class PerceiverCrossAttentionDecoder(nn.Module):
    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 embedder_out_dim: int,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 qk_norm=False):

        super().__init__()

        self.query_proj = nn.Linear(embedder_out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            qk_norm=qk_norm,
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)
        
        self.gradient_checkpointing = False

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor, embedder):
        queries = self.query_proj(embedder(queries))
        
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            ckpt_kwargs = {"use_reentrant": False}
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.cross_attn_decoder),
                queries,
                latents,
                **ckpt_kwargs,
            )
        else:
            x = self.cross_attn_decoder(queries, latents)

        x = self.ln_post(x)
        x = self.output_proj(x)
        return x


class MichelangeloAutoencoder(ModelMixin, ConfigMixin):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        n_samples: int = 4096,
        use_downsample: bool = False,
        num_latents: int = 256,
        point_feats: int = 0,
        embed_point_feats: bool = False,
        out_dim: int = 1,
        embed_dim: int = 64,
        embed_type: str = "fourier",
        num_freqs: int = 8,
        include_pi: bool = True,
        width: int = 768,
        heads: int = 12,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_ln_post: bool = False,
        use_flash: bool = False,
        use_multi_reso: Optional[bool] = False,
        resolutions: Optional[List[int]] = None,
        sampling_prob: Optional[List[float]] = None,
        qk_norm=False,
        **kwargs,
    ):
        super().__init__()
        
        self.embedder = get_embedder(embed_type=embed_type, num_freqs=num_freqs, include_pi=include_pi)
        # encoder
        self.init_scale = self.config.init_scale * math.sqrt(1.0 / self.config.width)
        
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.config.use_downsample,
            embedder_out_dim=self.embedder.out_dim,
            num_latents=self.config.num_latents,
            point_feats=self.config.point_feats,
            embed_point_feats=self.config.embed_point_feats,
            width=self.config.width,
            heads=self.config.heads,
            layers=self.config.num_encoder_layers,
            init_scale=self.init_scale,
            qkv_bias=self.config.qkv_bias,
            use_ln_post=self.config.use_ln_post,
            use_flash=self.config.use_flash,
            use_multi_reso=self.config.use_multi_reso,
            resolutions=self.config.resolutions,
            sampling_prob=self.config.sampling_prob,
            qk_norm=qk_norm,
        )

        if self.config.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.config.width, self.config.embed_dim * 2)
            self.post_kl = nn.Linear(self.config.embed_dim, self.config.width)
            self.latent_shape = (self.config.num_latents, self.config.embed_dim)
        else:
            self.latent_shape = (self.config.num_latents, self.config.width)

        self.transformer = Perceiver(
            n_ctx=self.config.num_latents,
            width=self.config.width,
            layers=self.config.num_decoder_layers,
            heads=self.config.heads,
            init_scale=self.config.init_scale,
            qkv_bias=self.config.qkv_bias,
            use_flash=self.config.use_flash,
            qk_norm=qk_norm,
        )

        # decoder
        self.decoder = PerceiverCrossAttentionDecoder(
            embedder_out_dim=self.embedder.out_dim,
            out_dim=self.config.out_dim,
            num_latents=self.config.num_latents,
            width=self.config.width,
            heads=self.config.heads,
            init_scale=self.config.init_scale,
            qkv_bias=self.config.qkv_bias,
            use_flash=self.config.use_flash,
            qk_norm=False,
        )
        
        self.gradient_checkpointing = False
    
    def encode(self,
               surface: torch.FloatTensor,
               sample_posterior: bool = True,
               random_fps=True):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        assert surface.shape[-1] == 3 + self.config.point_feats, f"\
            Expected {3 + self.config.point_feats} channels, got {surface.shape[-1]}"
        
        pc, feats = surface[..., :3], surface[..., 3:] # B, n_samples, 3
        bs, N, D = pc.shape
        if N > self.config.n_samples:
            # idx = furthest_point_sample(pc, self.config.n_samples) # (B, 3, npoint)
            # pc = gather_operation(pc, idx).transpose(2, 1).contiguous()
            # feats = gather_operation(feats, idx).transpose(2, 1).contiguous()
            flattened = pc.view(bs*N, D) # bs*N, 64
            batch = torch.arange(bs).to(pc.device)
            batch = torch.repeat_interleave(batch, N) # bs*N
            pos = flattened
            ratio = self.config.n_samples / N
            idx = fps(pos, batch, ratio=ratio)
            pc = pc.view(bs*N, -1)[idx].view(bs, -1, pc.shape[-1])
            feats = feats.view(bs*N, -1)[idx].view(bs, -1, feats.shape[-1])

        shape_latents = self.encoder(pc, feats, embedder=self.embedder, random_fps=random_fps) # B, num_latents, width
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)  # B, num_latents, embed_dim

        return shape_latents, kl_embed, posterior

    def decode(self, latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(latents) # [B, num_latents, embed_dim] -> [B, num_latents, width]

        return self.transformer(latents)

    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], occupancy logits
        """
        logits = self.decoder(queries, latents, embedder=self.embedder).squeeze(-1)

        return logits

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.config.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior
    
    def origin_forward(self,
                       surface: torch.FloatTensor,
                       queries: torch.FloatTensor,
                       sample_posterior: bool = True):
        shape_latents, kl_embed, posterior = self.encode(surface, sample_posterior=sample_posterior)

        latents = self.decode(kl_embed) # [B, num_latents, width]

        logits = self.query(queries, latents) # [B,]

        return shape_latents, latents, posterior, logits
    
    def forward(self,
                surface: torch.FloatTensor,
                sample_posterior: bool = True,
                random_fps=True):
        shape_latents, kl_embed, posterior = self.encode(surface, sample_posterior=sample_posterior, random_fps=random_fps)
        latents = self.decode(kl_embed) # [B, num_latents, width]
        return latents
    
    @torch.no_grad()
    def extract_geometry(self,
                         latents: torch.FloatTensor,
                         extract_mesh_func: str = "mc",
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         num_cells: int = 512,
                         num_chunks: int = 10000):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            num_cells=num_cells,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.query(batch_queries, latents)
            batch_logits.append(logits.cpu())

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy()

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                if extract_mesh_func == "mc":
                    from skimage import measure
                    vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0, method="lewiner")
                    # vertices, faces = mcubes.marching_cubes(grid_logits[i], 0)
                    vertices = vertices / grid_size * bbox_size + bbox_min
                    faces = faces[:, [2, 1, 0]]
                elif extract_mesh_func == "diffmc":
                    from diso import DiffMC
                    diffmc = DiffMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                else:
                    raise NotImplementedError(f"{extract_mesh_func} not implement")
                mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces.astype(np.int64))))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        return mesh_v_f, has_surface

    @staticmethod
    def generate_dense_grid_points(
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        num_cells: int,
        indexing: str = "ij"
    ):
        length = bbox_max - bbox_min
        x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
        y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
        z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
        [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
        xyz = np.stack((xs, ys, zs), axis=-1)
        xyz = xyz.reshape(-1, 3)
        grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

        return xyz, grid_size, length

    def _set_gradient_checkpointing(self, enable=True,gradient_checkpointing_func=None, Callable = None):

        for module in self.modules():
            if isinstance(module, Perceiver) or isinstance(module, PerceiverCrossAttentionEncoder) or isinstance(module, PerceiverCrossAttentionDecoder) :
                self._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
