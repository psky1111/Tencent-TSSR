import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import Timesteps,TimestepEmbedding

from src.custom_models.meshtron_opt.attention import MultiLevelRoPE
from src.custom_models.meshtron_opt.hourglass_transformer import HourglassCasualSlidingTransformerFlow
from src.custom_models.meshtron_opt.modules import DiscreteValueEmbeding
import torch.distributions as dist
import numpy as np

from collections import defaultdict

from tqdm import tqdm



def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )

    
### default value a=0 b =2 
class Kapplescheduler():
    order = 2
    def __call__(self, t):
        if self.order == 2:
            return t*t
        else:
            return t
    def derivate(self,t):
        if self.order == 2:
            return 2*t
        else:
            return 1

class Euler_sampler_h:
    def __init__(self,adaptable=True):
        self.adaptable = adaptable
        self.alpha = 12
        self.a = 2
        self.b = 0
        self.alpha_t = lambda t: 1 + (self.alpha * (t**self.a))
        self.beta_t = lambda t: self.alpha_t(t) - 1
    def adaptive_h(self,h,t,schduler:Kapplescheduler):
        t = t.clamp(1e-4,1)
        coeff = (1 - schduler(t))/schduler.derivate(t) - 1e-9
        h = torch.tensor([h],device=coeff.device)
        h_ada = torch.minimum(h,coeff)
        return h_ada
    def constant_h(self,h,t,schduler):
        return h
    def get_h(self,h,t,schduler):
        if self.adaptable:
            return self.adaptive_h(h,t,schduler)
        else:
            return self.constant_h(h,t,schduler)

class Euler_correct_sampler_h:
    def __init__(self,adaptable=True):
        self.adaptable = adaptable
        self.alpha = 12
        self.a = 0.25
        self.b = 0.25
        self.alpha_t = lambda t: 1 + (self.alpha * (t**self.a)*((1-self.b)**0.25))
        self.beta_t = lambda t: self.alpha_t(t) - 1
    def adaptive_h(self,h,t,schduler:Kapplescheduler):
        t = t.clamp(0+1e-3,1)
        coeff_left = self.alpha_t(t)*schduler.derivate(t)/(1-schduler(t))
        coeff_right = self.beta_t(t)*schduler.derivate(t)/schduler(t)
        coeff = coeff_left + coeff_right
        coeff = 1/coeff - 1e-9
        h = torch.tensor([h],device=coeff.device)
        h_ada = torch.minimum(h,coeff)
        return h_ada
    def constant_h(self,h,t,schduler):
        return h
    def get_h(self,h,t,schduler):
        if self.adaptable:
            return self.adaptive_h(h,t,schduler)
        else:
            return self.constant_h(h,t,schduler)
        


class MeshtronNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        dim=768,
        context_dim=768,
        n_heads=8,
        n_kv_heads=None,
        depth=None,
        shorten_factor=3,
        updown_sample_type='linear',
        theta_rotary_emb=1000000.0,
        vocab_size=None,
        n_sliding_triangles=None,
        ffn_multiple=4,
        qk_norm=None,
        condition_interval=4,
        max_infer_triangles=64000,
        num_timesteps=1000,
        time_freq_dim=768,
        codebook_size=128,
        noise_type="mask",
        RoPE="1d",
        Alibi=False,
        metric_induced=False,
        **kwargs,
    ):
        super().__init__()

        if dim % n_heads != 0:
            raise ValueError(f"dim: {dim} must be divisible by heads: {n_heads}")
        head_dim = dim // n_heads
        
        if n_kv_heads is not None and context_dim is not None and n_kv_heads * head_dim < context_dim:
            raise ValueError(f"context info dim: {context_dim} cannot be reduced: {n_kv_heads * head_dim}")
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.face_count_emb = DiscreteValueEmbeding(embedding_dim=context_dim)


        recursive_depth = [
            depth[0] // 2, 
            [depth[1] // 2, depth[2], depth[1] // 2], 
            depth[0] // 2
        ]
        

        window_len = n_sliding_triangles * 9  
        self.noise_type = noise_type
        self.connect_loss = None

        max_token_num = max_infer_triangles * 9

        self.freqs_cis =  MultiLevelRoPE(head_dim, max_token_num)  
        self.alibi = None
        
        self.hourglass_transformer = HourglassCasualSlidingTransformerFlow(
            dim=dim,
            context_dim=context_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            depth=recursive_depth,
            shorten_factor=shorten_factor,
            updown_sample_type=updown_sample_type,
            max_window_len=window_len,
            ffn_multiple=ffn_multiple,
            qk_norm=qk_norm,
            condition_interval=condition_interval,
        )
        
        # diffusion component
        self.scheduler = torch.linspace(
            1 / num_timesteps, 1, steps=num_timesteps)
        #self.scheduler = torch.linspace(0,1,steps=1/num_timesteps)
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=time_freq_dim)
        self.time_proj = nn.Linear(time_freq_dim, context_dim)
        self.pad_token_id = codebook_size + 2
        self.eos_token_id = codebook_size + 1
        self.bos_token_id = codebook_size 
        self.kapper = Kapplescheduler()
        self.sampler_h = Euler_sampler_h()
        self.null_condition =  nn.Parameter(
            torch.randn(1, 3072, context_dim)
        )
        if noise_type == "mask":
            self.mask_token_id = codebook_size 
            self.true_codebook_size = codebook_size + 1
            #self.config.codebook_size = codebook_size +1
            
        self.to_logits = nn.Linear(dim, self.config.codebook_size)

        self.to_acc = torch.nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.SiLU(),
            nn.Linear(dim//2, dim//4),
            nn.SiLU(),
            nn.Linear(dim//4, 1),
            )


        self.gradient_checkpointing = False
        self.metric_induced = metric_induced

        self.mask_mode = torch.tensor([1]).long()
        self.no_mask_mode = torch.tensor([0]).long()
    
    def generate_sampling_noise_scheduler(self,N:int):
        T_train = self.scheduler.size(0)
        linear_progress = np.arange(0, N + 1) / N
        #ideal_t_points = 0.5 * (1 - np.cos(linear_progress * np.pi))
        ideal_t_points = linear_progress
        #ideal_t_points = np.power(np.arange(0, N + 1) / N, 2)
        unfiltered_indices = np.floor(ideal_t_points * T_train)
        unfiltered_indices = np.minimum(unfiltered_indices, T_train).astype(int)
        time_indices = np.unique(unfiltered_indices)
        h_values = np.diff(time_indices) / T_train
        time_indices = torch.tensor(time_indices,device=self.device,dtype=torch.long)
        h_values = torch.tensor(h_values,device=self.device)
        return time_indices, h_values

    def beta(self, t):
        t = t.clamp(0,1-1e-4)
        c = 100
        a = 3
        beta_t = c*(t/(1-t))**a
        return beta_t
    
    def beta_derivative(self,t):
        c = 100
        a = 3
        beta_t_derivative = c*a*(t/(1-t))**(a-1)/(1-t)**2
        return beta_t_derivative

    def norm(self,x):
        V = self.config.codebook_size
        return ((x/V)*2) - 1
    def add_noise_metric(self, x_1, t):
        B, S = x_1.shape
        V = self.config.codebook_size
        if isinstance(t, float):
            t = torch.tensor([t],device=x_1.device)
            # --- 1. Create the Coordinate Tensors ---
            # The ground truth tokens, reshaped for broadcasting
        x_1_coords = x_1.view(B, S, 1).float() # Shape: [B, S, 1]
            
            # A tensor representing all possible token values [0, 1, ..., V-1]
        vocab_coords = torch.arange(V, device=x_1.device).view(1, 1, V).float() # Shape: [1, 1, V]
            
            # --- 2. THE CRITICAL STEP: Calculate Distance in Coordinate Space ---
            # Broadcasting: [B, S, 1] - [1, 1, V] -> [B, S, V]
            # This calculates the difference between each token in x_1 and every
            # possible token in the vocabulary.
        distance_matrix_signed = self.norm(x_1_coords) - self.norm(vocab_coords)
        distance_matrix_squared = distance_matrix_signed**2 # This is our d(x, x_1)

            # --- THE FIX IS HERE ---
            # Get beta_t from the new, controllable scheduler.
        beta_t = self.beta(t).view(B, 1, 1)

        logits_t = -beta_t * distance_matrix_squared
        probs_t = torch.nn.functional.softmax(logits_t, dim=-1)

            # Sample x_t from the distribution
        #probs_t_reshaped = probs_t.view(B * S, V)
        #x_t_flat = torch.multinomial(probs_t_reshaped, num_samples=1)
        #x_t = x_t_flat.view(B, S)
        dist = torch.distributions.Categorical(probs_t)
        x_t = dist.sample()
        return x_t
    
    def forward_hidden(self, input_ids, context, t, cache_kv=False, cache_position=None, is_sampling=False, segment_position=0,
                face_count_id=None):
        # 将 cache 清理工作放在 forward 函数内部，无需在外部调用。避免 validation 开启 cache 后忘记关闭，对训练产生影响。
        # face_count_id now is the mode flag
        if not cache_kv:
            self.hourglass_transformer.clean_kv_cache()
        if not is_sampling:
            # 在当前逻辑下，routing_cache 一定只有在采样阶段才会开启
            self.hourglass_transformer.clean_routing_cache()

            
            
        if face_count_id is not None:
            face_count_embed = self.face_count_emb(face_count_id).unsqueeze(1)  # shape: (b, 1, c)
            context = torch.concat([context, face_count_embed], dim=1)
       
        x = self.token_emb(input_ids)

        if isinstance(self.freqs_cis,MultiLevelRoPE):
            with torch.no_grad():
                self.freqs_cis.token_level_freq = self.freqs_cis("token",seq_len=input_ids.shape[1],device=x.device)
                self.freqs_cis.vetrix_level_freq =self.freqs_cis("vertex",seq_len=input_ids.shape[1]//3,device=x.device)
                self.freqs_cis.face_level_freq = self.freqs_cis("face",seq_len=input_ids.shape[1]//9,device=x.device)

        else:
            if self.freqs_cis.device != x.device:
                self.freqs_cis = self.freqs_cis.to(device=x.device)
        if self.alibi is not None:
            with torch.no_grad():
                self.alibi.token_level_alibi = self.alibi("token",seq_len=input_ids.shape[1],device=x.device).to(x.dtype)
                self.alibi.vertex_level_alibi = self.alibi("vertex",seq_len=input_ids.shape[1]//3,device=x.device).to(x.dtype)
                self.alibi.face_level_alibi = self.alibi("face",seq_len=input_ids.shape[1]//9,device=x.device).to(x.dtype)
        
        t_proj = self.timesteps_proj(t).type_as(x)
        t_emb = self.time_embedder(t_proj).type_as(x)
        t_emb = self.time_proj(t_emb).unsqueeze(1)
        context = torch.concat([context, t_emb], dim=1)
        #x = x
        
        hidden_states = self.hourglass_transformer(
            x, 
            context=context,
            context_level=x,
            cache_kv=cache_kv,
            cache_position=cache_position,
            is_sampling=is_sampling,
            segment_position=segment_position,
            freqs_cis=self.freqs_cis,
            alibi=self.alibi,
        )
        #logits = self.to_logits(hidden_states)
        return hidden_states
    
    def x2prob(self,x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Converts discrete token indices to a probability distribution.
        
        If temperature is high, the distribution is "soft" (closer to uniform).
        If temperature is very low, it approaches a one-hot distribution.
        """
        # Create one-hot vectors
        one_hot = torch.nn.functional.one_hot(x, num_classes=self.true_codebook_size).float()
        
        # If temperature is 1.0, this is just the one-hot distribution.
        if temperature == 1.0:
            return one_hot
            
        # Create "logits" by scaling the one-hot. The correct token gets a high
        # positive value, and all others get a small negative value.
        # This ensures softmax(logits) is a soft peak at the correct token.
        logits = torch.log(one_hot + 1e-9) # Add epsilon for numerical stability
        
        # Apply temperature to the logits.
        # A higher temperature "flattens" the final softmax distribution.
        soft_probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        
        return soft_probs
    
    def sample_p(self,pt,mode="logit"):
        B,S,V = pt.shape
        pt = pt.reshape(-1,V)
        xt = torch.multinomial(pt,1)
        return xt.reshape(B,S)
    
    def forward_u(self,t,xt,model_output):
        t = t.clamp(1e-3,1)
        dirac_xt = self.x2prob(xt)
        p1t = torch.softmax(model_output,dim=-1)
        kappa_coeff = self.kapper.derivate(t)/ (1 - self.kapper(t))
        return kappa_coeff * (p1t - dirac_xt)
    
    def forward_u_metric(self,t,xt:torch.Tensor,model_output:torch.Tensor):
        beta_derivative = self.beta_derivative(t)
        x_1 = model_output.softmax(dim=-1).argmax(dim=-1)
        B, S = x_1.shape
        V = self.config.codebook_size
        if isinstance(t, float):
            t = torch.tensor([t],device=x_1.device)
            # --- 1. Create the Coordinate Tensors ---
            # The ground truth tokens, reshaped for broadcasting
        x_1_coords = x_1.view(B, S, 1).float() # Shape: [B, S, 1]
            
            # A tensor representing all possible token values [0, 1, ..., V-1]
        vocab_coords = torch.arange(V, device=x_1.device).view(1, 1, V).float() # Shape: [1, 1, V]
            
            # --- 2. THE CRITICAL STEP: Calculate Distance in Coordinate Space ---
            # Broadcasting: [B, S, 1] - [1, 1, V] -> [B, S, V]
            # This calculates the difference between each token in x_1 and every
            # possible token in the vocabulary.
        distance_matrix_signed = self.norm(x_1_coords) - self.norm(vocab_coords)
        distance_matrix_squared = distance_matrix_signed**2 # This is our d(x, x_1)

            # --- THE FIX IS HERE ---
            # Get beta_t from the new, controllable scheduler.
        beta_t = self.beta(t).view(B, 1, 1)

        logits_t = -beta_t * distance_matrix_squared
        probs_t = torch.nn.functional.softmax(logits_t, dim=-1)

        distance_matrix_x1t_z = ((self.norm(xt) - self.norm(x_1))**2).unsqueeze(-1)
        distance_term = (distance_matrix_x1t_z - distance_matrix_squared).relu()
        u = probs_t*distance_term*beta_derivative
        return u

    
    def backward_u(self,t,xt,x_noise,model_output):
        t = t.clamp(1e-3,1)
        dirac_xt = self.x2prob(xt)
        #x0 = torch.zeros_like(xt) + self.mask_token_id
        #x0 = x0.long()
        p0t = torch.ones_like(model_output).softmax(dim=-1)
        #diract_noise = self.x2prob(x_noise)
        #kapper_t = self.kapper(t)
        #p0t = (1-kapper_t)*diract_noise + kapper_t*dirac_xt
        kappa_coeff = self.kapper.derivate(t)/ (self.kapper(t))
        return kappa_coeff * (dirac_xt - p0t)
    
    def correct_u(self,t,xt,x_noise,model_output,alpha_t, beta_t):
        return alpha_t*self.forward_u(t,xt,model_output) - beta_t*self.backward_u(t,xt,x_noise,model_output)

    
    def add_noise(
        self,
        x_1: torch.Tensor, 
        x_0:torch.Tensor,
        t: torch.Tensor, 
        max_temp=10.,
        min_temp=1.0
    ) -> torch.Tensor:
        """
        Calculates the conditional path probability p_t(x | x_1) for a metric-induced path.

        This function does NOT compute the marginal p_t(x_t). It computes the tractable
        conditional distribution needed to sample x_t during training.

        Args:
            x_1 (torch.Tensor): The ground-truth target tokens (the condition). Shape: [B, S].
            x_0: The pure nose tokens
            t (torch.Tensor): The time, sampled per batch item. Shape: [B, 1, 1].

        Returns:
            torch.Tensor: The conditional probabilities p_t(x | x_1). Shape: [B, S, V].
        """
        if isinstance(t, float):
            t = torch.tensor([t],device=x_1.device)
        t = t.clamp(0,1-1e-3) # avoid NAN
        temperature = max_temp * (1.0 - t) + min_temp * t
        p_0 = self.x2prob(x_0)
        p_1 = self.x2prob(x_1) # x1 should be the full mask 
        #p_u = torch.ones_like(p_1) / self.config.codebook_size
        
        kapper_t = self.kapper(t)
        p_t = (1-kapper_t)*p_0 + kapper_t*p_1
        noised_x = self.sample_p(p_t,mode="prob")
        return noised_x
    
    def forward(self, input_ids, context, timestep, **kwargs):
        v_pred = self.forward_hidden(input_ids, context,timestep, **kwargs)
        logits = self.to_logits(v_pred)
        acc = self.to_acc(v_pred)
        return logits, acc
    
    def forward_classifier(self, input_ids, context, timestep, **kwargs):
        with torch.no_grad():
            v_pred = self.forward_hidden(input_ids, context,timestep, **kwargs)
        acc = self.to_acc(v_pred.detach())
        return acc
    
    def forward_noise(self,x:torch.Tensor,t:torch.Tensor,should_noise:torch.Tensor | None,eps=1e-3)->torch.Tensor:
        if (x == self.eos_token_id).any():
            x = x[:,9:-9] # remove special token
        keep_mask = x == self.bos_token_id
        keep_mask |= x == self.pad_token_id
        if self.noise_type == "mask":
            full_x = torch.full_like(x,self.mask_token_id)
        else:
            full_x = torch.randint_like(x,self.config.codebook_size)
        if self.metric_induced:
            noise_x = self.add_noise_metric(x, t).detach()
        else:
            noise_x = self.add_noise(x,full_x,t).detach()
        noise_x = noise_x * (~keep_mask) + x * keep_mask
        keep_mask = x == noise_x
        return noise_x, keep_mask, full_x
    
    def _get_sampling_timesteps(self, num_sampling_steps):
        return torch.linspace(
            len(self.scheduler) - 1,
            len(self.scheduler) // num_sampling_steps,
            num_sampling_steps,
            device=self.device,
            dtype=torch.long,
        )
    
    def forward_random_face(self,input_ids):
        if (input_ids == self.eos_token_id).any():
            input_ids = input_ids[:,9:-9] # remove special token
        b,s = input_ids.shape
        #min_length, max_length = -10, 50
        #perturb_length = s + 9*torch.randint(min_length, max_length + 1, (1,)).item()
        perturb_length = s
        perturbed_sequence = torch.full((b, perturb_length), self.mask_token_id, 
                                   dtype=input_ids.dtype, device=input_ids.device)
        keep_mask = torch.full((b, perturb_length), False, dtype=torch.bool, device=input_ids.device)
        return perturbed_sequence, keep_mask, perturb_length
    
    def sample_with_timesteps(self,context,t, num_sampling_steps: int,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        x: torch.Tensor | None = None,
        stochasticity: float = 0.,
        yield_intermediate: bool = False,
        yield_logits: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        input_ids: torch.Tensor | None=None,
        face_count_id=None,
        **kwargs):
        #randon_noise = torch.randint_like(input_ids,self.config.codebook_size)
        #noise_x, keep_mask,_ = self.forward_noise(input_ids, t, should_noise=None)
        noise_x, keep_mask,seq_leg = self.forward_random_face(input_ids)
        return self.sample(context, num_sampling_steps,
        num_samples= num_samples,
        sequence_length = seq_leg,
        x = noise_x,
        stochasticity = stochasticity,
        yield_intermediate = yield_intermediate,
        yield_logits = yield_logits,
        temperature= temperature,
        face_count_id=face_count_id,
        cfg_scale = cfg_scale,keep_mask=keep_mask,
        input_ids = input_ids,
        **kwargs)
    
    def dynamic_guidance_throttling(
        self,
        x_t,
        scale_t, # The κ'_t / (1 - κ_t) term for this timestep
        p_t_con, # The conditional posterior p(x_1|x_t, c)
        p_t_uncon, # The unconditional posterior p(x_1|x_t)
        w, # The *maximum* guidance scale (e.g., 15.0)
        temperature_coeffect=1.0 # The temperature for the posteriors
    ):
        epsilon = 1e-9 # For numerical stability

        # 1. Get Conditional and Unconditional Velocities
        u_con = self.forward_u(scale_t, x_t, p_t_con / temperature_coeffect)
        u_uncon = self.forward_u(scale_t, x_t, p_t_uncon / temperature_coeffect)

        # 2. Decompose into Inflow Components
        # The inflow is simply the positive part of the velocity field.

        u_ts = u_uncon + w*(u_con - u_uncon)

        return u_ts
  
    def sample(self,context, num_sampling_steps: int,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        x: torch.Tensor | None = None,
        stochasticity: float = 0.,
        yield_intermediate: bool = False,
        yield_logits: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        face_count_id = None,
        enable_bad_model=False,
        choice="max_confident",
        keep_mask=None,
        return_inter=True,
        input_ids=None,
        **kwargs):
        device = context.device
        if x is None:
            if self.noise_type == "mask":
                x_t = torch.full((num_samples,sequence_length),self.mask_token_id,device=device)
                #x_t[:,0:9] = self.bos_token_id
                keep_mask = x_t != self.mask_token_id
            else:
                x_t = torch.randint(0,self.config.codebook_size,(num_samples,sequence_length),device=device)
                #x_t[:,0:9] = self.bos_token_id
                keep_mask = x_t == self.bos_token_id
        else:
            x_t = x
            if self.noise_type == "mask":
                keep_mask = x_t != self.mask_token_id
            else:
                keep_mask = keep_mask

        time_idx, hs = self.generate_sampling_noise_scheduler(num_sampling_steps)
        inter_x = []
        no_mask_mode = self.no_mask_mode.to(device=x_t.device)
        mask_mode = self.mask_mode.to(device=x_t.device)
        mask_logits_scheduler = torch.linspace(0.5,0.9,num_sampling_steps,device=device)

        for step in tqdm(range(len(hs)), desc="Sampling"):
            #t_val = step * h
            t = time_idx[step].unsqueeze(0)
            input_x = x_t
            u_ts, _ = self.forward(input_x, context, t,face_count_id=mask_mode)
            scaler = t/1000
            scaler = max(scaler,1e-5)
            x_t = u_ts.softmax(dim=-1).argmax(dim=-1)
            u_ts, _ = self.forward(x_t, context, t,face_count_id=no_mask_mode)
            pred_acc = self.forward_classifier(x_t, context, t,face_count_id=no_mask_mode)
            pred_acc = pred_acc.sigmoid().squeeze(dim=-1) # [0,1]
            x_t = u_ts.softmax(dim=-1).argmax(dim=-1)
            inter_x.append(x_t.detach().cpu())

            pred_mask = pred_acc < 0.5
            pred_mask = pred_mask.long()

            x_t = pred_mask * self.mask_token_id + (1 - pred_mask) * x_t

        if return_inter:
            return inter_x
        return inter_x[-1]
    
    def clean_cache(self):
        self.hourglass_transformer.clean_kv_cache()
        self.hourglass_transformer.clean_routing_cache()

    def prepare_cache(self, config=None):
        self.hourglass_transformer.prepare_kv_cache(config)
    
    def reorder_cache(self, all_chosen_beam_idx):
        self.hourglass_transformer.reorder_kv_cache(all_chosen_beam_idx)
        self.hourglass_transformer.reorder_routing_cache(all_chosen_beam_idx)

    def _set_gradient_checkpointing(self, enable=True,gradient_checkpointing_func=None, Callable = None):
        #if hasattr(module, "gradient_checkpointing"):
        #    module.gradient_checkpointing = value
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                self._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
