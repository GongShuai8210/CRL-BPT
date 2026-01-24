"""
BBPT-UCL: Black-Box Prompt Tuning with Unified Curriculum Learning

This version combines PPO loss and Curriculum loss into a SINGLE unified loss:

    L_total = L_ppo + λ_ref(t)·L_reference + λ_div(t)·L_diversity + λ_con(t)·L_contrastive

Key advantages:
1. Single optimizer, single backward pass
2. Gradients from both losses combined naturally
3. No conflict between separate optimizers
4. More direct curriculum learning signal

The curriculum weights λ(t) automatically adjust based on training progress t.

Control Variate Reward Strategy:
    R = r_π - α*(r_base - μ_base)
Where α* = Cov(r_π, r_base) / Var(r_base) is the optimal control coefficient.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb
import utils_r
from utils_r import ControlVariateConfig, ControlVariateRewardComputer, create_control_variate_reward_computer
from peft import LoraConfig

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# DassL imports
from dassl.data import DataManager
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed
import custom_datasets.oxford_pets
import custom_datasets.oxford_flowers
import custom_datasets.fgvc_aircraft
import custom_datasets.dtd
import custom_datasets.eurosat
import custom_datasets.stanford_cars
import custom_datasets.food101
import custom_datasets.sun397
import custom_datasets.caltech101
import custom_datasets.ucf101
import custom_datasets.imagenet
import custom_datasets.imagenet_sketch
import custom_datasets.imagenetv2
import custom_datasets.imagenet_a
import custom_datasets.imagenet_r
import custom_datasets.svhn
import custom_datasets.clevr
import custom_datasets.resisc45



import clip
import ast

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
os.environ["HF_TOKEN"] = 'your_huggingface_token_here'  
# =============================================================================
# Unified PPO + Curriculum Trainer
# =============================================================================
class LossScaler:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.ema = {}

    def scale(self, name, loss):
        if name not in self.ema:
            self.ema[name] = loss.detach().abs()
        else:
            self.ema[name] = self.beta * self.ema[name] + (1 - self.beta) * loss.detach().abs()

        # 用 EMA 缩放，而不是当前值
        return loss / (self.ema[name] + 1e-6)
def set_all_seeds(seed: int):
  
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass

@dataclass
class UCLConfig:
    """Configuration for Unified Curriculum Learning."""
    # PPO hyperparameters
    learning_rate: float = 5e-6
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    kl_coef: float = 0.1
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Curriculum weights (base values, will be modulated by progress)
    base_reference_weight: float = 0.3
    base_diversity_weight: float = 0.2
    base_contrastive_weight: float = 0.5

    # Curriculum schedule
    reference_decay_rate: float = 3.0
    diversity_growth_rate: float = 3.0
    contrastive_peak: float = 0.5
    contrastive_width: float = 0.3

    # Loss parameters
    contrastive_margin: float = 0.1
    similarity_threshold: float = 0.3
    top_k_ratio: float = 0.5


class UnifiedPPOCurriculumTrainer:
    """
    Unified trainer that combines PPO and Curriculum losses.

    Total Loss = L_ppo + λ_ref(t)·L_reference + λ_div(t)·L_diversity + λ_con(t)·L_contrastive

    Where:
    - L_ppo: Standard PPO policy gradient loss with clipping
    - L_reference: Encourages similarity to good reference prompts
    - L_diversity: Encourages novel patterns, penalizes repetition
    - L_contrastive: Margin ranking loss between good and bad prompts
    - λ(t): Curriculum weights that change with training progress
    """

    def __init__(
            self,
            e_a,
            model: nn.Module,
            tokenizer,
            config: UCLConfig,
            reference_prompts: List[str],
            total_steps: int,
            device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.total_steps = max(1, total_steps)

        # Reference prompts for curriculum
        self.reference_prompts = reference_prompts
        self.reference_patterns = [self._extract_pattern(p) for p in reference_prompts]

        # Single optimizer for everything
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

        # Progress tracking
        self.current_step = 0
        self.current_progress = 0.0
        self.e_a = e_a
        # Pattern tracking for diversity
        self.pattern_counts = {}
        self.recent_patterns = deque(maxlen=100)

        # Value function baseline (for advantage estimation)
        self.value_baseline = deque(maxlen=100)

        self.reference_prompt_words = [
            self._extract_pattern(p).lower().split()
            for p in reference_prompts
        ]

        self.recent_prompts = []
        self.max_recent = 100

        # Statistics
        self.loss_history = {
            'total': deque(maxlen=100),
            'ppo': deque(maxlen=100),
            'reference': deque(maxlen=100),
            'diversity': deque(maxlen=100),
            'contrastive': deque(maxlen=100),
        }

    def _lcs_length(self, s1: List[str], s2: List[str]) -> int:
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0

        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]

    def _compute_rouge_l(self, candidate: List[str], reference: List[str]) -> float:
        """计算 ROUGE-L F1"""
        if len(candidate) == 0 or len(reference) == 0:
            return 0.0

        lcs_len = self._lcs_length(candidate, reference)

        precision = lcs_len / len(candidate)
        recall = lcs_len / len(reference)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
    def _extract_pattern(self, prompt: str) -> str:
        """Extract structural pattern from prompt."""
        pattern = prompt.lower().strip()
        pattern = pattern.replace('{}', '<CLS>')
        pattern = ' '.join(pattern.split())
        return pattern

    def _get_ngrams(self, text: str, n: int = 2) -> set:
        """Get word n-grams."""
        words = text.split()
        if len(words) < n:
            return {text}
        return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}



    def _compute_novelty(self, prompt: str) -> Tuple[float, bool]:
 
        pattern = self._extract_pattern(prompt)
        words = pattern.lower().split()

        if len(words) == 0:
            return 1.0, False

        max_similarity = 0.0
        for recent_words in self.recent_prompts:
            sim = self._compute_rouge_l(words, recent_words)
            max_similarity = max(max_similarity, sim)

        self.recent_prompts.append(words)
        if len(self.recent_prompts) > self.max_recent:
            self.recent_prompts.pop(0)

        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1

        novelty = 1.0 - max_similarity
        is_repeated = max_similarity > 0.8 

        return novelty, is_repeated

    def _compute_reference_similarity(self, prompt: str) -> float:
       
        pattern = self._extract_pattern(prompt)
        words = pattern.lower().split()

        if len(words) == 0:
            return 0.0

        max_similarity = 0.0
        for ref_words in self.reference_prompt_words:
            sim = self._compute_rouge_l(words, ref_words)
            max_similarity = max(max_similarity, sim)

        return max_similarity

    def _get_curriculum_weights(self):
      
        e_a = self.e_a
        t = self.current_progress
        s_ref = np.exp(-e_a * t)  
        s_con = np.exp(-((t - 0.5) / 0.3) ** 2)  
        s_div = 1 - np.exp(-e_a * t)  

        total = (s_ref +  s_div)  

        return {
            'reference': s_ref / total,
            'diversity': s_div / total
        }

    def _compute_log_probs_and_values(
            self,
            query_tensor: torch.Tensor,
            response_tensors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities, entropy, and values for responses.

        Returns:
            log_probs: Log probability of each response
            entropies: Entropy of each response
            values: Value estimates for each response
        """
        all_log_probs = []
        all_entropies = []
        all_values = []

        for response in response_tensors:
            if response.numel() == 0:
                all_log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                all_entropies.append(torch.tensor(0.0, device=self.device))
                all_values.append(torch.tensor(0.0, device=self.device))
                continue

            # Concatenate query and response
            input_ids = torch.cat([query_tensor, response], dim=-1).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids)

            # Handle tuple output from ValueHead model
            if isinstance(outputs, tuple):
                logits = outputs[0].float()
                values = outputs[2].float() if len(outputs) > 2 else torch.zeros(1, device=self.device)
            else:
                logits = outputs.logits.float()
                values = torch.zeros(1, device=self.device)

            query_len = query_tensor.size(0)
            response_len = response.size(0)

            # Get logits for response tokens
            response_logits = logits[0, query_len - 1:query_len - 1 + response_len, :]
            log_probs_all = F.log_softmax(response_logits, dim=-1)

            # Get log prob of actual tokens
            response_tokens = response.to(self.device)
            token_log_probs = log_probs_all.gather(
                dim=-1,
                index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            seq_log_prob = token_log_probs.sum()
            all_log_probs.append(seq_log_prob)

            # Entropy
            probs = F.softmax(response_logits, dim=-1)
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()
            all_entropies.append(entropy)

            # Value (average over response)
            if isinstance(values, torch.Tensor) and values.numel() > 1:
                value = values[0, query_len - 1:query_len - 1 + response_len].mean()
            else:
                value = values.mean() if isinstance(values, torch.Tensor) else torch.tensor(0.0, device=self.device)
            all_values.append(value)

        return torch.stack(all_log_probs), torch.stack(all_entropies), torch.stack(all_values)

    def compute_ppo_loss(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            advantages: torch.Tensor,
            values: torch.Tensor,
            returns: torch.Tensor,
            entropies: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with clipping.

        L_ppo = L_policy + vf_coef * L_value - entropy_coef * entropy
        """
        # Policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total PPO loss
        ppo_loss = (
                policy_loss +
                self.config.vf_coef * value_loss +
                self.config.entropy_coef * entropy_loss
        )

        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropies.mean().item(),
            'ratio_mean': ratio.mean().item(),
        }

        return ppo_loss, stats

    def compute_reference_loss(
            self,
            prompts: List[str],
            log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        
        similarities = []
        for prompt in prompts:
            sim = self._compute_reference_similarity(prompt)
            similarities.append(sim)

        sim_tensor = torch.tensor(similarities, device=self.device, dtype=torch.float32)

        loss = -(sim_tensor * log_probs).mean()

        return loss, sim_tensor.mean().item()

    def compute_diversity_loss(
            self,
            prompts: List[str],
            log_probs: torch.Tensor,
            entropies: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
       
        novelty_scores = []
        is_repeated_list = []

        for prompt in prompts:
            novelty, is_repeated = self._compute_novelty(prompt)
            novelty_scores.append(novelty)
            is_repeated_list.append(is_repeated)

        novelty_tensor = torch.tensor(novelty_scores, device=self.device, dtype=torch.float32)
        repeated_tensor = torch.tensor(is_repeated_list, device=self.device, dtype=torch.float32)

        novelty_loss = -(novelty_tensor * log_probs).mean()

        if repeated_tensor.sum() > 0:
            repetition_penalty = (repeated_tensor * log_probs).mean()
        else:
            repetition_penalty = torch.tensor(0.0, device=self.device)

        total_loss = novelty_loss + repetition_penalty

        penalty = 1.0 - novelty_tensor

        diversity_loss = (penalty * log_probs).mean()

        stats = {
            'mean_novelty': novelty_tensor.mean().item(),
            'mean_penalty': penalty.mean().item(),
        }

        return diversity_loss, stats

    def compute_contrastive_loss(
            self,
            log_probs: torch.Tensor,
            rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Contrastive Loss: Margin ranking between good and bad prompts.
        L_con = Σ max(0, margin - (log π(good) - log π(bad)))
        """
        n = len(rewards)
        k = max(1, int(n * self.config.top_k_ratio))

        sorted_indices = torch.argsort(rewards, descending=True)
        good_indices = sorted_indices[:k]
        bad_indices = sorted_indices[-k:]

        good_log_probs = log_probs[good_indices]
        bad_log_probs = log_probs[bad_indices]

        # Pairwise margin loss
        losses = []
        for g_lp in good_log_probs:
            for b_lp in bad_log_probs:
                margin_loss = F.relu(self.config.contrastive_margin - (g_lp - b_lp))
                losses.append(margin_loss)

        if losses:
            loss = torch.stack(losses).mean()
            margin_diff = (good_log_probs.mean() - bad_log_probs.mean()).item()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            margin_diff = 0.0

        return loss, margin_diff

    @torch.no_grad()
    def generate(
            self,
            query_tensor: torch.Tensor,
            generation_kwargs: dict,
            num_return_sequences: int = 4
    ) -> List[torch.Tensor]:
        """Generate responses."""
        self.model.eval()

        outputs = self.model.generate(
            query_tensor.unsqueeze(0),
            **generation_kwargs,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
        )

        # Extract only the new tokens (remove the prompt)
        prompt_len = query_tensor.size(0)
        response_tensors = []
        for seq in outputs.sequences:
            response = seq[prompt_len:]
            response_tensors.append(response)

        self.model.train()
        return response_tensors

    def step(
            self,
            query_tensor: torch.Tensor,
            response_tensors: List[torch.Tensor],
            prompts: List[str],
            rewards: List[float]
    ) -> Dict[str, float]:
        """
        Perform unified PPO + Curriculum optimization step.

        This computes ALL losses together and does a SINGLE backward pass.
        """
        self.current_step += 1
        self.current_progress = min(1.0, self.current_step / self.total_steps)

        # Get curriculum weights
        weights = self._get_curriculum_weights()

        self.model.train()

        # Compute log probs, entropy, values (with gradients)
        log_probs, entropies, values = self._compute_log_probs_and_values(
            query_tensor, response_tensors
        )

        # Detach for old log probs (PPO uses ratio of new/old)
        old_log_probs = log_probs.detach()

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Compute advantages and returns
        # Simple advantage: reward - baseline
        baseline = np.mean(list(self.value_baseline)) if self.value_baseline else rewards_tensor.mean().item()
        self.value_baseline.extend(rewards)

        advantages = rewards_tensor - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = rewards_tensor  # For single-step, returns = rewards

        # =====================================================================
        # Compute all losses
        # =====================================================================

        # 1. PPO Loss
        ppo_loss, ppo_stats = self.compute_ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropies
        )
        epsilon = 1e-8
        # 2. Reference Loss (curriculum)
        ref_loss, mean_similarity = self.compute_reference_loss(prompts, log_probs)

        # 3. Diversity Loss (curriculum)
        div_loss, div_stats = self.compute_diversity_loss(prompts, log_probs, entropies)

        # 4. Contrastive Loss (curriculum)
        # con_loss, margin_diff = self.compute_contrastive_loss(log_probs, rewards_tensor)

        # =====================================================================
        # UNIFIED TOTAL LOSS
        # =====================================================================
        scaler = LossScaler()
        total_loss = (
                scaler.scale('ppo', ppo_loss) +
                weights['reference'] * scaler.scale('ref', ref_loss) +
                weights['diversity'] * scaler.scale('div', div_loss)
        )
        # total_loss = (
        #          ppo_loss +
        #         weights['reference'] * ref_loss +
        #         weights['diversity'] * div_loss
        # )

        # =====================================================================
        # Single backward pass and optimization step
        # =====================================================================
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # Record statistics
        # self.loss_history['total'].append(total_loss.item())
        # self.loss_history['ppo'].append(ppo_loss.item())
        # self.loss_history['reference'].append(ref_loss.item())
        # self.loss_history['diversity'].append(div_loss.item())
        # self.loss_history['contrastive'].append(con_loss.item())
        self.loss_history['total'].append(total_loss.item())
        self.loss_history['ppo'].append(ppo_loss.item())
        self.loss_history['reference'].append(ref_loss.item())
        self.loss_history['diversity'].append(div_loss.item())
        # self.loss_history['contrastive'].append(con_loss.item())

        return {
            # Total
            'loss/total': total_loss.item(),

            # PPO components
            'loss/ppo': ppo_loss.item(),
            'ppo/policy_loss': ppo_stats['policy_loss'],
            'ppo/value_loss': ppo_stats['value_loss'],
            'ppo/entropy': ppo_stats['entropy'],
            'ppo/ratio_mean': ppo_stats['ratio_mean'],

            # Curriculum components
            'loss/reference': ref_loss.item(),
            'loss/diversity': div_loss.item(),
            # 'loss/contrastive': con_loss.item(),

            # Curriculum weights
            'curriculum/weight_reference': weights['reference'],
            'curriculum/weight_diversity': weights['diversity'],
            'curriculum/progress': self.current_progress,

            # Curriculum stats
            'curriculum/mean_similarity': mean_similarity,
            'curriculum/mean_novelty': div_stats['mean_novelty'],
            'curriculum/num_repeated': div_stats['mean_penalty'],
            # 'curriculum/margin_diff': margin_diff,
            'curriculum/unique_patterns': len(self.pattern_counts),

            # Training
            'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'train/advantages_mean': advantages.mean().item(),
            'train/advantages_std': advantages.std().item(),
        }

    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            'total_steps': self.current_step,
            'final_progress': self.current_progress,
            'unique_patterns': len(self.pattern_counts),
            'avg_losses': {
                k: np.mean(list(v)) if v else 0
                for k, v in self.loss_history.items()
            },
            'top_patterns': sorted(
                self.pattern_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# =============================================================================
# Argument Parser
# =============================================================================

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_model', type=str, default='Qwen/Qwen2.5-3B-Instruct') # Qwen/Qwen2.5-3B-Instruct google/gemma-2b-it
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/16')
    parser.add_argument('--dataset', type=str, default='svhn')
    parser.add_argument('--dataset_config_file', type=str, default='/your_own _path/datasets_config')
    parser.add_argument('--data_root', type=str, default='/your_own _path/data')
    parser.add_argument('--cache_dir', type=str, default='/your_own _path/LLMs/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_prompt_length', type=int, default=15)
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--meta_prompt', type=str,
                        default='''You are an expert in prompt engineering for Vision-Language Models. Your goal is to write a text description (prompt) that accurately classifies images from the {} dataset.
                            ''')
    parser.add_argument('--prompt_per_image', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output')

    # Human prompt examples
    parser.add_argument('--num_human_examples', type=int, default=10)
    parser.add_argument('--human_example_style', type=str, default='detailed',
                        choices=['default', 'detailed', 'simple'])
    parser.add_argument('--num_human_examples_dir', type=str, default='prompt_examples/')
    parser.add_argument('--use_top_examples', action='store_true', default=True)
    parser.add_argument('--dynamic_examples', action='store_true', default=False)
    parser.add_argument('--max_num_apis', type=int, default=2000)
    parser.add_argument('--eva_step', type=int, default=10)

    # ==========================================================================
    # Unified PPO + Curriculum Learning Parameters
    # ==========================================================================
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for unified optimizer')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--kl_coef', type=float, default=0.1,
                        help='KL penalty coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy bonus coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Curriculum base weights
    parser.add_argument('--base_reference_weight', type=float, default=0.3,
                        help='Base weight for reference loss')
    parser.add_argument('--base_diversity_weight', type=float, default=0.2,
                        help='Base weight for diversity loss')
    parser.add_argument('--base_contrastive_weight', type=float, default=0.5,
                        help='Base weight for contrastive loss')

    # Curriculum schedule
    parser.add_argument('--reference_decay_rate', type=float, default=3.0,
                        help='How fast reference weight decays')
    parser.add_argument('--diversity_growth_rate', type=float, default=3.0,
                        help='How fast diversity weight grows')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--e_a', type=float, default=2.0)

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # ==========================================================================
    # Control Variate Reward Parameters
    # ==========================================================================
    parser.add_argument('--baseline_prompt', type=str, default='a photo of a {}.',
                        help='Baseline prompt template for control variate')
    parser.add_argument('--cv_ema_beta', type=float, default=0.99,
                        help='EMA decay factor for online estimation of variance/covariance')
    parser.add_argument('--cv_acc_weight', type=float, default=30.0,
                        help='Weight for accuracy in reward computation')
    parser.add_argument('--cv_softmax_diff_weight', type=float, default=10.0,
                        help='Weight for softmax difference in reward')
    parser.add_argument('--cv_warmup_steps', type=int, default=20,
                        help='Steps before using adaptive alpha (use alpha=1 during warmup)')
    parser.add_argument('--cv_calibration_batches', type=int, default=50,
                        help='Number of batches for baseline calibration (computing mu_base)')
    parser.add_argument('--repetition_penalty1', type=float, default=1.2,
                        help='repetition_penalty of tokens')

    return parser.parse_args()

# =============================================================================
# Setup Functions
# =============================================================================

def setup_dassl_cfg(args):
    """Setup DassL configuration for data loading."""
    cfg = get_cfg_default()
    cfg.DATASET.ROOT = args.data_root
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size * 4
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = 'bicubic'
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
    cfg.SEED = args.seed
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.freeze()
    return cfg
def load_clip_model(backbone_name, device):
    """Load CLIP model as target model."""
    model, preprocess = clip.load(backbone_name, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    from datetime import datetime, timedelta, timezone
    beijing_tz = timezone(timedelta(hours=8))
    start_time = datetime.now(beijing_tz)
    str_start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=f'bbpt_vlm_{args.dataset}',
        config=vars(args),
        name=f'{args.dataset}_{args.num_shots}shot_cv_reward'
    )

    # --- Data Setup ---
    print("Setting up DassL data manager...")
    cfg = setup_dassl_cfg(args)
    cfg.merge_from_file(os.path.join(args.dataset_config_file, f"{args.dataset}.yaml"))
    set_all_seeds(cfg.SEED)

    args.HUMAN_PROMPT_EXAMPLES_file = os.path.join(args.num_human_examples_dir, args.dataset)
    with open(args.HUMAN_PROMPT_EXAMPLES_file, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find('[')
    end = content.rfind(']') + 1
    list_string = content[start:end]
    HUMAN_PROMPT_EXAMPLES = ast.literal_eval(list_string)

    args.meta_prompt = args.meta_prompt.format(args.dataset)

    dm = DataManager(cfg)

    classnames = dm.dataset.classnames
    train_loader = dm.train_loader_x
    val_loader = dm.val_loader
    test_loader = dm.test_loader

    reference_prompts = [ex['prompt'] for ex in
                         sorted(HUMAN_PROMPT_EXAMPLES, key=lambda x: x['accuracy'], reverse=True)[:20]]
    best_ref_prompt = [ex['prompt'] for ex in
                            sorted(HUMAN_PROMPT_EXAMPLES, key=lambda x: x['accuracy'], reverse=True)[:1]][0]

    print(f'Dataset: {args.dataset}, Shots: {args.num_shots}')

    # --- CLIP Setup ---
    print(f"Loading CLIP (backbone: {args.clip_backbone})...")
    clip_model, clip_preprocess = load_clip_model(args.clip_backbone, device)

    # ==========================================================================
    # Initialize Control Variate Reward Computer
    # ==========================================================================
    print("\nInitializing Control Variate Reward Strategy...")

    cv_reward_computer = create_control_variate_reward_computer(
        clip_model=clip_model,
        classnames=classnames,
        device=device,
        baseline_prompt=args.baseline_prompt,
        ema_beta=args.cv_ema_beta,
        acc_weight=args.cv_acc_weight,
        softmax_diff_weight=args.cv_softmax_diff_weight,
        warmup_steps=args.cv_warmup_steps
    )

    # Calibrate baseline BEFORE training (computes μ_base)
    calibration_stats = cv_reward_computer.calibrate_baseline(
        data_loader=train_loader,
        num_batches=args.cv_calibration_batches,
        show_progress=True
    )

    wandb.log({
        'calibration/mu_base': calibration_stats['mu_base'],
        'calibration/std_base': calibration_stats['std_base'],
        'calibration/mean_accuracy': calibration_stats['mean_accuracy'],
    })

    # --- Estimate total steps ---
    steps_per_epoch = len(train_loader)
    total_steps = min(args.max_num_apis // args.prompt_per_image, steps_per_epoch * args.epochs)

    # --- Load Agent Model ---
    print(f"Loading agent model: {args.agent_model}...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model, cache_dir=args.cache_dir)
    agent_tokenizer.pad_token = agent_tokenizer.eos_token

    agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        peft_config=lora_config,
        cache_dir=args.cache_dir
    )

    # --- Initialize Unified Trainer ---

    ucl_config = UCLConfig(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        kl_coef=args.kl_coef,
        vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        base_reference_weight=args.base_reference_weight,
        base_diversity_weight=args.base_diversity_weight,
        base_contrastive_weight=args.base_contrastive_weight,
        reference_decay_rate=args.reference_decay_rate,
        diversity_growth_rate=args.diversity_growth_rate,
    )

    trainer = UnifiedPPOCurriculumTrainer(
        e_a=args.e_a,
        model=agent_model,
        tokenizer=agent_tokenizer,
        config=ucl_config,
        reference_prompts=reference_prompts,
        total_steps=total_steps,
        device=device
    )

    print(f"\n{'=' * 60}")
    print("Training Configuration with Control Variate Reward")
    print(f"{'=' * 60}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  PPO Clip Range: {args.clip_range}")
    print(f"  Base Reference Weight: {args.base_reference_weight}")
    print(f"  Base Diversity Weight: {args.base_diversity_weight}")
    print(f"  Base Contrastive Weight: {args.base_contrastive_weight}")
    print(f"  Reference Prompts: {len(reference_prompts)}")
    print(f"  Estimated Total Steps: {total_steps}")
    print(f"\n  --- Control Variate Settings ---")
    print(f"  Baseline Prompt: \"{args.baseline_prompt}\"")
    print(f"  μ_base (global mean reward): {calibration_stats['mu_base']:.4f}")
    print(f"  EMA Beta: {args.cv_ema_beta}")
    print(f"  Warmup Steps: {args.cv_warmup_steps}")
    print(f"{'=' * 60}\n")
    
    def get_forbidden_token_ids(tokenizer):
        """
        Retrieve all token IDs containing Chinese characters or specific special symbols.
        """
        forbidden_ids = []

        # Define the list of special characters to ban
        # Note: '\\' represents the backslash itself, '\n' represents a newline
        special_chars_to_ban = []

        print("Scanning vocabulary to ban specific characters...")
        for token_id in range(tokenizer.vocab_size):
            # Decode the token; skip_special_tokens=False ensures we can check all content
            token = tokenizer.decode([token_id])

            # 1. Check if it contains Chinese characters
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in token)

            # 2. Check if it contains specific special characters
            is_special_bad = any(char in token for char in special_chars_to_ban)

            if is_chinese or is_special_bad:
                forbidden_ids.append(token_id)

        print(f"Total banned tokens: {len(forbidden_ids)}")
        return torch.tensor(forbidden_ids)

    # Get all forbidden IDs
    bad_token_ids = get_forbidden_token_ids(agent_tokenizer)
    # 
    # chinese_token_ids = get_chinese_token_ids(agent_tokenizer)
    # Generation kwargs
    generation_kwargs = {
        "top_k": 0,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "do_sample": True,
        "pad_token_id": agent_tokenizer.eos_token_id,
        "max_new_tokens": args.max_prompt_length,
        "min_length": 2,
        "repetition_penalty": args.repetition_penalty1,      
        "no_repeat_ngram_size": 3,         
        "bad_words_ids": [[tid] for tid in bad_token_ids.tolist()],
    }

    queue = utils_r.TopAccuracyTextsNoDuplicates(max_size=5)
    best_val_prompts = []
    max_best_prompts = 5

    print("\nStarting BBPT training with Control Variate Reward Strategy...")

    step_id = -1
    api_num = 0
    reach_max_num_apis = False

    import json
    saved_prompts = {}

    for ep in tqdm(range(args.epochs)):
        for batch in train_loader:
            step_id += 1
            api_num += args.prompt_per_image

            if api_num % 20 == 0:
                print(f"\033[1;32mapi_num: {api_num}\033[0m")


            if api_num > args.max_num_apis:
                reach_max_num_apis = True
                break

            images = batch['img'].to(device)
            labels = batch['label'].to(device)

            # --- Generation Phase ---
            with torch.no_grad():
                query_messages = utils_r.build_agent_messages_with_examples(
                    HUMAN_PROMPT_EXAMPLES,
                    meta_prompt=args.meta_prompt,
                    num_human_examples=args.num_human_examples,
                    human_example_style=args.human_example_style,
                    use_top_examples=args.use_top_examples
                )
                query_encoded = agent_tokenizer.apply_chat_template(
                    query_messages,
                    return_tensors='pt',
                    add_generation_prompt=True  
                ).view(-1).to(device)


                try:
                    response_tensors = trainer.generate(
                        query_encoded, generation_kwargs,
                        num_return_sequences=args.prompt_per_image
                    )
                except RuntimeError as e:
                    print(f"\033[1;31mWarning: Generation failed ({e}), skipping batch\033[0m")
                    continue

                used_prompts = [agent_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                for r in response_tensors]

            if sum([len(p) for p in used_prompts]) < args.prompt_per_image * 10:
                continue

            # =================================================================
            # Compute Stabilized Rewards using Control Variate
            # R = r_π - α*(r_base - μ_base)
            # =================================================================
            rewards, accuracies, cv_stats = cv_reward_computer.compute_stabilized_rewards_batch(
                images, labels, used_prompts
            )

            # --- Unified Training Step ---
            stats = trainer.step(
                query_tensor=query_encoded,
                response_tensors=response_tensors,
                prompts=used_prompts,
                rewards=rewards  # Use stabilized rewards
            )

            # --- Logging ---
            for i in range(len(rewards)):
                print(f'Stabilized R: {rewards[i]:.2f}, Acc: {accuracies[i]:.4f}, '
                      f'Prompt: {used_prompts[i][:60]}...')
                queue.add(rewards[i], used_prompts[i], ep)

            log_dict = {
                'mean_reward': np.mean(rewards),
                'mean_raw_reward': cv_stats['mean_raw_reward'],
                'mean_accuracy': np.mean(accuracies),
                'reward_std': np.std(rewards),
                # Control Variate specific logs
                'cv/alpha': cv_stats['alpha'],
                'cv/batch_bias': cv_stats['batch_bias'],
                'cv/r_base': cv_stats['r_base'],
                'cv/accuracy_base': cv_stats['accuracy_base'],
                'cv/var_base_ema': cv_stats['var_base_ema'],
                'cv/cov_cross_ema': cv_stats['cov_cross_ema'],
            }
            log_dict.update(stats)

            wandb.log(log_dict)

            ### save  generated prompt of specific api_nums

            if api_num % 20 == 0:
                print("generated prompt...")
                response_tensors = trainer.generate(
                    query_encoded, generation_kwargs, num_return_sequences=2
                )
                eval_prompts = [agent_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                for r in response_tensors]
                print(f" Generated prompts for test: {eval_prompts}")
                saved_prompts[api_num] = eval_prompts[0]


            # --- Evaluation Phase ---
            if step_id % args.eva_step == 0:
                cv_overall_stats = cv_reward_computer.get_statistics()

                print(f"\n--- Evaluation at step {step_id} ---")
                print(f"  Progress: {stats['curriculum/progress']:.1%}")
                print(f"  Control Variate α: {cv_stats['alpha']:.4f} "
                      f"(mean: {cv_overall_stats['mean_alpha']:.4f})")
                print(f"  Batch Bias: {cv_stats['batch_bias']:.4f}")
                print(f"  Var(r_base) EMA: {cv_stats['var_base_ema']:.4f}")
                print(f"  Cov(r_π, r_base) EMA: {cv_stats['cov_cross_ema']:.4f}")
                print(f"  Curriculum Weights - Ref: {stats['curriculum/weight_reference']:.3f}, "
                      f"Div: {stats['curriculum/weight_diversity']:.3f} ")
                print(f"  Unique Patterns: {stats['curriculum/unique_patterns']}")
                print(f"  Total Loss: {stats['loss/total']:.4f} "
                      f"(PPO: {stats['loss/ppo']:.4f}, "
                      f"Ref: {stats['loss/reference']:.4f}, "
                      f"Div: {stats['loss/diversity']:.4f} ")
                      # f"Con: {stats['loss/contrastive']:.4f})")

                try:
                    response_tensors = trainer.generate(
                        query_encoded, generation_kwargs, num_return_sequences=2
                    )
                except RuntimeError:
                    continue

                eval_prompts = [agent_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                for r in response_tensors]
                print(f"  Generated prompts: {eval_prompts}")

                val_accs = utils_r.evaluate_clip_on_loader(
                    clip_model, val_loader, eval_prompts, classnames, device
                )
                print(f'  Validation acc: {val_accs}')

                prompt_queue = queue.get_top_texts()
                if len(prompt_queue) > 0:
                    candidate_prompts = [p[1] for p in prompt_queue]
                    val_accs_queue = utils_r.evaluate_clip_on_loader(
                        clip_model, val_loader, candidate_prompts, classnames, device
                    )

                    for prompt, val_acc_i in zip(candidate_prompts, val_accs_queue):
                        existing_prompts = [p[1] for p in best_val_prompts]
                        if prompt in existing_prompts:
                            continue

                        if len(best_val_prompts) < max_best_prompts:
                            best_val_prompts.append((val_acc_i, prompt, ep, step_id))
                        elif val_acc_i > best_val_prompts[-1][0]:
                            best_val_prompts.append((val_acc_i, prompt, ep, step_id))

                        best_val_prompts.sort(key=lambda x: x[0], reverse=True)
                        best_val_prompts = best_val_prompts[:max_best_prompts]
                    best_prompt_file = f"{str_start_time}_best_prompts.txt"
                    with open(os.path.join(args.output_dir, best_prompt_file), 'w') as f:
                        for rank, (v_acc, p, e, s) in enumerate(best_val_prompts, 1):
                            f.write(f'Rank {rank} | Val: {v_acc:.4f} | Ep: {e}, Step: {s}\n{p}\n\n')

                if args.dynamic_examples and ep != 0:
                    utils_r.update_examples_from_queue(queue, top_k=2)

                wandb.log({
                    'eval/val_acc': np.mean(val_accs),
                    'best_val_acc': best_val_prompts[0][0] if best_val_prompts else 0,
                    'cv/mean_alpha_overall': cv_overall_stats['mean_alpha'],
                    'cv/std_alpha_overall': cv_overall_stats['std_alpha'],
                })

        if reach_max_num_apis:
            break
    with open("generated_prompts.json", "w", encoding="utf-8") as f:
        json.dump(saved_prompts, f, indent=2, ensure_ascii=False)
    # --- Final Evaluation ---
    print('\n' + '=' * 60)
    print('Final Test Evaluation')
    print('=' * 60)

    summary = trainer.get_summary()
    cv_final_stats = cv_reward_computer.get_statistics()

    print(f"\n--- Training Summary ---")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Unique Patterns: {summary['unique_patterns']}")
    print(f"  Average Losses: {summary['avg_losses']}")
    print(f"\n--- Control Variate Summary ---")
    print(f"  Mean α: {cv_final_stats['mean_alpha']:.4f}")
    print(f"  Std α: {cv_final_stats['std_alpha']:.4f}")
    print(f"  Final Var(r_base) EMA: {cv_final_stats['var_base_ema']:.4f}")
    print(f"  Final Cov(r_π, r_base) EMA: {cv_final_stats['cov_cross_ema']:.4f}")
    print(f"\n  Top Patterns:")
    for pattern, count in summary['top_patterns']:
        print(f"    [{count}x] {pattern[:50]}...")

    if best_val_prompts:
        best_ref_prompt = (best_val_prompts[0][0],best_ref_prompt,0,0)
        best_val_prompts.append(best_ref_prompt)

        print(f'\nTesting Top {len(best_val_prompts)} Validation Prompts...')
        test_accs = utils_r.evaluate_clip_on_loader(
            clip_model, test_loader, [p[1] for p in best_val_prompts], classnames, device
        )
        for i, acc in enumerate(test_accs):
            print(f"Rank {i + 1} Test Acc: {acc:.4f} (Val: {best_val_prompts[i][0]:.4f})")
            print(f"  Prompt: {best_val_prompts[i][1]}")

        wandb.log({'final_best_test_acc': test_accs[0]})

    # Save results
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Method: Unified PPO + Curriculum + Control Variate Reward\n\n')
        f.write(f'Control Variate Settings:\n')
        f.write(f'  Baseline Prompt: {args.baseline_prompt}\n')
        f.write(f'  μ_base: {calibration_stats["mu_base"]:.4f}\n')
        f.write(f'  Mean α: {cv_final_stats["mean_alpha"]:.4f}\n\n')
        f.write(f'Training Summary:\n')
        f.write(f'  Unique Patterns: {summary["unique_patterns"]}\n')
        f.write(f'  Avg Losses: {summary["avg_losses"]}\n\n')
        f.write('Top Prompts:\n')
        for i, (val_acc, prompt, ep, step) in enumerate(best_val_prompts):
            f.write(f'{i + 1}. {prompt}\n')
            f.write(f'   Val Acc: {val_acc:.4f}\n\n')

    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()

