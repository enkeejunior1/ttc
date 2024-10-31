########################
# SEDD/Loss, Scheduler #
########################
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import einsum
from typing import Union


class Loss(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def forward(self, output, t=None, xt=None, x0=None, attention_mask=None, **kwargs):
        """
        TODO: need to verify the code
        """
        logits = self.scheduler.output_to_logits(output, xt)
        attn_mask  = torch.ones_like(x0).bool() if attention_mask is None else attention_mask # TODO
        trans_mask = torch.logical_and(x0 != self.scheduler.mask_idx, xt == self.scheduler.mask_idx)
        token_mask = torch.logical_and(attn_mask, trans_mask)

        assert len(t.shape) == 1
        sigma_bar_t = self.scheduler.sigma_bar(t)
        sigma_t = self.scheduler.sigma(t)
        
        log_p_theta = torch.gather(
            input=logits.log_softmax(dim=-1), dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        
        loss = -log_p_theta * (sigma_t / torch.expm1(sigma_bar_t))[:, None]
        nlls = loss[token_mask]
        count = token_mask.sum()
        if count == 0:
            warnings.warn("Warning: there are no tokens for training. Zero flip.")
            return 0
        else:
            return nlls.sum() / count


class Scheduler(nn.Module):
    """
    We only care about masked diffusion models

    Train 
        1. t, samples -> sigma (alphas_comprod)  - (sample_transition) -> noisy_samples
        2. pred_score = model(samples, t)
        3. score = get_score(samples, noisy_samples)
        4. loss_weight = get_loss_weight(t)
        5. loss = loss_weight * comp_loss(pred_score, score)
        
    Sampling
    """
    def __init__(
        self, args
    ):  
        super().__init__()

        # basic configs
        self.num_vocabs = args.num_vocabs + 1 # "absorb"
        self.length = args.length
        self.eps = args.eps
        self.model_name = args.model_name
        self.mask_idx = args.num_vocabs
        
        # init noise schedule (similar to alphas_cumprod)
        if args.noise_schedule == "loglinear":
            self.sigma_bar = lambda t: -torch.log1p(-(1 - self.eps) * t) # page 15
            self.sigma = lambda t: (1 - self.eps) / (1 - (1 - self.eps) * t) # sigma_bar / dt
        
    def add_noise(
        self, samples: torch.LongTensor, t: Union[int, torch.LongTensor], generator=None, 
    ):
        '''x0 -> xt'''
        # snr
        sigma_bar = self.sigma_bar(t)
        
        # perturb samples (absorb)
        perturb_prob = 1 - (-sigma_bar).exp()
        perturbed_samples = torch.where(
            torch.rand(*samples.shape, device=samples.device, generator=generator) < perturb_prob[:, None],
            self.mask_idx, samples
        )
        return perturbed_samples
    
    def output_to_logits(self, output, xt=None, t=None):
        if self.model_name == 'mdlm': 
            # ref: https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py#L261
            # _subs_parameterization
            assert xt is not None
            logits = output

            # prob = logits.exp()
            logits[:, :, self.mask_idx] = -torch.inf
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

            # prob(xt[unmasked indices] = xt) = 1
            unmasked_indices = (xt != self.mask_idx)
            logits[unmasked_indices] = -torch.inf
            logits[unmasked_indices, xt[unmasked_indices]] = 0
        else:
            raise ValueError(f'invalid model_name: {self.model_name}')
        return logits
    
    def sample_latent(self, num_samples):
        return (self.num_vocabs-1) * torch.ones(num_samples, self.length).long()

    def step(self, output, xt, t, step_size):
        pass
    
    @torch.no_grad()
    def euler_sample(self, model, xt, t, s, euler_steps=1, sample_steps=5):
        '''xt -> xu, with euler sampling'''
        assert euler_steps == 1, 'notimplemented error for euler_steps > 1'
        assert (t > s).all(), f'{t}, {s}'
        assert (s > 0).all(), f'{s}, {self.eps}'
        timesteps = torch.linspace(1, self.eps, sample_steps+1, device=xt.device)
        timesteps = (t[:, None] - s[:, None]) * timesteps[None, :] + s[:, None]
        noises = []
        xt_traj = []
        for i in range(sample_steps):
            dt = timesteps[:, i] - timesteps[:, i+1]
            curr_t = timesteps[:, i]

            sigma_bar_t = self.sigma_bar(curr_t)
            output = model(xt, sigma_x=sigma_bar_t)
            output = self.step(output, xt, curr_t, dt)
            xt = output.xt
            
            xt_traj.append(xt)
            noises.append(output.noise)
        xt_traj = torch.stack(xt_traj, dim=1)
        noises = torch.stack(noises, dim=1)
        return xt, xt_traj, noises, timesteps[:, 1:]


class SchedulerOutput:
    def __init__(self, xt, xt_prob=None, x0_prob=None, rev_rate=None, tau=None, noise=None):
        self.xt = xt
        self.tau = tau
        self.x0_prob = x0_prob
        self.xt_prob = xt_prob
        self.rev_rate = rev_rate
        self.noise = noise


class EulerScheduler(Scheduler):
    def step(self, output, xt, t, step_size, generator=None, if_last=False, **kwargs):
        sigma_t = self.sigma_bar(t)
        sigma_s = self.sigma_bar(t - step_size)

        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape

        alpha_t = 1 - torch.exp(-sigma_t)
        alpha_s = 1 - torch.exp(-sigma_s)
        alpha_t = alpha_t[:, None, None]
        alpha_s = alpha_s[:, None, None]

        logits = self.output_to_logits(output, xt)
        q_xs = logits.exp() * (alpha_t - alpha_s)
        q_xs[:, :, self.mask_idx] = alpha_s[:, :, 0] if not if_last else -torch.inf
        xs, noise = sample_categorical(q_xs)

        copy_flag = (xt != self.mask_idx).to(xt.dtype)
        xs = copy_flag * xt + (1 - copy_flag) * xs
        return SchedulerOutput(xs, xt_prob=q_xs, x0_prob=logits.softmax(dim=-1), noise=noise)

def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1), gumbel_noise

##############
# SEDD/model #
##############

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, sigma, **kwargs):
        
        if indices.dtype == torch.long:
            x = self.vocab_embed(indices)
        else:
            x = F.linear(indices, self.vocab_embed.embedding.T, bias=None)

        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            x = self.output_layer(x, c)


        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
        # x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        if indices.dtype == torch.long:
            x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        else:
            x = torch.scatter(x, -1, indices.argmax(dim=-1)[..., None], torch.zeros_like(x[..., :1]))

        return x
    

import torch
from torch import nn


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )


# @torch.jit.script
def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
    # try:
    #     import flash_attn.layers.rotary
    #     cos = cos[0,:,0,0,:cos.shape[-1]//2]
    #     sin = sin[0,:,0,0,:sin.shape[-1]//2]
    #     return flash_attn.layers.apply_rotary_emb_qkv_(
    #         qkv, cos, sin
    #     )
    # except:
    return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)
    

import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return modulate(x, shift, scale)

import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma)
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn