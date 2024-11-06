# This loss function code is heavily based on SEDD implementation.
# ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import einsum
from typing import Union

class Loss(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def forward(self, input_ids=None, labels=None, conds=None, **kwargs):
        """
        TODO: need to verify the code
        """
        # prepare model input
        x0 = input_ids
        t = (1 - self.scheduler.eps) * torch.rand(x0.shape[0], device=x0.device) + self.scheduler.eps
        sigma_bar_t = self.scheduler.sigma_bar(t)
        sigma_t = self.scheduler.sigma(t)
        xt = self.scheduler.add_noise(x0, t)
        xt[:, :conds.shape[1]][conds] = x0[:, :conds.shape[1]][conds]

        # model forward
        output = self.model(xt, torch.zeros_like(sigma_bar_t))

        # MDLM loss
        logits = self.scheduler.output_to_logits(output, xt)
        attn_mask  = torch.ones_like(x0).bool() if labels is None else labels # TODO
        trans_mask = torch.logical_and(x0 != self.scheduler.mask_idx, xt == self.scheduler.mask_idx)
        token_mask = torch.logical_and(attn_mask, trans_mask)

        assert len(t.shape) == 1
        log_p_theta = torch.gather(
            input=logits, # input=logits.log_softmax(dim=-1), 
            dim=-1, index=x0[:, :, None],
        ).squeeze(-1)
        
        loss = -log_p_theta * (sigma_t / torch.expm1(sigma_bar_t))[:, None]
        nlls = loss[token_mask]
        count = token_mask.sum()
        if count == 0:
            warnings.warn("Warning: there are no tokens for training. Zero flip.")
            loss = 0
            token_accuracy = 0
        else:
            loss = nlls.sum() / count
            token_accuracy = (logits.argmax(dim=-1) == x0)[token_mask].float().mean().item()
        return LossOutput(loss=loss, token_accuracy=token_accuracy)


class LossOutput:
    def __init__(self, loss=None, token_accuracy=None, **kwargs):
        self.loss = loss
        self.token_accuracy = token_accuracy


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
        self.eps = 1e-5
        self.model_name = args.base_model
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
    def euler_sample(self, model, xt, t, s, num_inference_steps=5):
        '''xt -> xu, with euler sampling'''
        t=t * torch.ones(xt.shape[0], device=xt.device) if isinstance(t, float) else t
        s=s * torch.ones(xt.shape[0], device=xt.device) if isinstance(s, float) else s
        assert (t > s).all(), f'{t}, {s}'
        assert (s > 0).all(), f'{s}, {self.eps}'
        timesteps = torch.linspace(1, self.eps, num_inference_steps+1, device=xt.device)
        timesteps = (t[:, None] - s[:, None]) * timesteps[None, :] + s[:, None]
        for i in range(num_inference_steps):
            dt = timesteps[:, i] - timesteps[:, i+1]
            curr_t = timesteps[:, i]

            sigma_bar_t = self.sigma_bar(curr_t)
            output = model(xt, torch.zeros_like(sigma_bar_t))
            output = self.step(output, xt, curr_t, dt)
            xt = output.xt
        return xt
    
    @torch.no_grad()
    def maskgit_sample(self, model, xt, num_inference_steps=5):
        '''xt -> xu, with euler sampling'''
        length = (xt == self.mask_idx).sum(dim=0)

        eps = 1e-3
        t = torch.linspace(1, eps, num_inference_steps + 1)
        k = (1 - (-self.sigma_bar(t)).exp()) * length
        k = k.long()
        k[-1] = 0

        for i in range(num_inference_steps):
            dk = k[i] - k[i+1]
            sigma_bar_t = self.sigma_bar(k[None, i])
            output = model(xt, torch.zeros_like(sigma_bar_t))
            output = self.step(output, xt, dk)
            xt = output.xt
        return xt


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
    

class MaskGITScheduler(Scheduler):
    def step(self, output, xt, step_size, generator=None, if_last=False, **kwargs):
        # generate x0 ~ p_x0
        logits = self.output_to_logits(output, xt)
        p_x0 = logits.exp()
        p_x0[:, :, self.mask_idx] = -torch.inf
        x0, noise = sample_categorical(p_x0)

        # mask x0 w.r.t confidence 
        conf = torch.gather(p_x0, -1, x0)
        conf[x0 != self.mask_idx] = -torch.inf
        conf_v, _ = torch.topk(conf, step_size, dim=-1)
        mask = (conf - conf_v[None, None, :]).to(xt.dtype)
        xs = mask * xt + (1 - mask) * x0
        return SchedulerOutput(xs, xt_prob=None, x0_prob=p_x0.softmax(dim=-1), noise=noise)
    

class GillespieScheduler(EulerScheduler):
    def add_noise(
        self, samples: torch.FloatTensor, k: Union[int, torch.LongTensor], generator=None, 
    ):
        '''0 -> 1 -> ... -> K'''
        token_prob = torch.rand(*samples.shape, device=samples.device, generator=generator)
        if samples.shape[1] - k != 0:
            values, idx = torch.topk(token_prob, samples.shape[1] - k, dim=-1, largest=False)
            t = (values.max(dim=-1).values / (1-self.eps)).clamp(min=0, max=1)
            perturbed_samples = torch.scatter(samples, -1, idx, self.mask_idx)
        else:
            t = torch.zeros(samples.shape[0], device=samples.device) + 1e-5
            perturbed_samples = samples
        return perturbed_samples, t
    
    
    def logits_to_score():
        return 
    
    
    def step(self, output, xt, t, dk, rev_rate=None, generator=None, if_last=False):
        '''Algorithm 1 from https://arxiv.org/abs/2407.21243'''
        if rev_rate is None:
            sigma = self.sigma(t)
            logits = self.output_to_logits(output)
            score = self.logits_to_score(logits)
            rev_rate = sigma[..., None, None] * self.Q_tilde(xt, score)

        # sample holding time
        r = rev_rate[..., :-1]
        tau = sample_exponential(r.sum(dim=-1), generator=generator)

        # sample token 
        tau, idx = torch.topk(tau, dk, dim=-1, largest=False)
        r = torch.gather(r, 1, idx[..., None].repeat(1,1,r.size(-1)))
        r = r / r.sum(dim=-1, keepdim=True)
        xt = torch.scatter(xt, -1, idx, sample_categorical(r, generator=generator))
        return SchedulerOutput(xt, rev_rate=rev_rate, tau=tau.max(dim=-1).values)
    

# ===========================================================================================
# ===========================================================================================
# ===========================================================================================




class AnalyticScheduler(Scheduler):
    def staggered_score(self, score, delta_sigma_bar):
        '''
        TODO need to understand
        ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L234
        '''
        extra_const = (1 - (delta_sigma_bar[:, None]).exp()) * score.sum(dim=-1)
        score *= delta_sigma_bar[:, None, None].exp()
        score[..., -1] += extra_const
        return score
    
    def transp_transition(self, i, sigma):
        '''
        TODO need to understand
        ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L218
        '''
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.num_vocabs)
        edge += torch.where(
            i == self.mask_idx,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge
    
    def step(self, output, xt, t, step_size, generator=None, if_last=False, **kwargs):
        curr_sigma_bar = self.sigma_bar(t)
        next_sigma_bar = self.sigma_bar(t - step_size)
        delta_sigma_bar = curr_sigma_bar - next_sigma_bar
        score = self.output_to_score(output)

        stag_score = self.staggered_score(score, delta_sigma_bar)
        probs = stag_score * self.transp_transition(xt, delta_sigma_bar)
        probs = probs[..., :-1] if if_last else probs
        xt = sample_categorical(probs, generator=generator)
        return SchedulerOutput(xt)
    
    
def sample_exponential(lambda_, eps=1e-6, generator=None):
    if generator is None:
        exp_noise = torch.rand_like(lambda_)
    else:
        exp_noise = torch.rand(lambda_.shape, generator=generator, device=generator.device).to(lambda_)
    return -1 / (lambda_ + eps) * torch.log(eps + (1 - eps) * exp_noise)

def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1), gumbel_noise

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)