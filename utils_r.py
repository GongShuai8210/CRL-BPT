"""
Utility Functions for BBPT-VLM (Improved Version)

This version includes solutions for sparse and noisy reward signals:
1. Multi-batch reward aggregation for stability
2. Reward normalization (rank-based and z-score)
3. Running baseline for variance reduction
4. Adaptive reward weighting
5. Confidence-aware reward computation

Following the coding style of the original BBPT utils.py,
adapted for Vision-Language Models (CLIP as target).
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import heapq
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Optional
from collections import deque

import clip


# =============================================================================
# SOLUTION 1: Stable Reward Computer with Multi-Batch Aggregation
# =============================================================================

class StableRewardComputer:
    """
    Computes stable rewards by aggregating over multiple batches and applying
    various normalization techniques to reduce variance.

    Key features:
    - Multi-batch evaluation for more reliable accuracy estimates
    - Running baseline subtraction for variance reduction
    - Reward normalization (z-score or rank-based)
    - Confidence-weighted rewards
    """

    def __init__(
        self,
        clip_model,
        classnames: List[str],
        device: torch.device,
        baseline_window: int = 50,
        use_rank_normalization: bool = False,
        use_zscore_normalization: bool = True,
        confidence_weight: float = 0.3,
        acc_weight: float = 30.0,
        margin_weight: float = 10.0
    ):
        """
        Args:
            clip_model: Frozen CLIP model for evaluation
            classnames: List of class names for the dataset
            device: Device to use for computation
            baseline_window: Window size for running baseline
            use_rank_normalization: Whether to use rank-based normalization
            use_zscore_normalization: Whether to use z-score normalization
            confidence_weight: Weight for confidence bonus in reward
            acc_weight: Weight for accuracy component
            margin_weight: Weight for softmax margin component
        """
        self.clip_model = clip_model
        self.classnames = classnames
        self.device = device

        # Normalization settings
        self.use_rank_normalization = use_rank_normalization
        self.use_zscore_normalization = use_zscore_normalization

        # Running statistics for baseline
        self.baseline_window = baseline_window
        self.reward_history = deque(maxlen=baseline_window)
        self.running_mean = 0.0
        self.running_std = 1.0

        # Reward weights
        self.confidence_weight = confidence_weight
        self.acc_weight = acc_weight
        self.margin_weight = margin_weight

        # Cache for text features (avoid recomputation)
        self._text_feature_cache = {}

    def _get_text_features(self, prompt: str) -> torch.Tensor:
        """Get or compute text features for a prompt (with caching)."""
        if prompt in self._text_feature_cache:
            return self._text_feature_cache[prompt]

        # Build text prompts for all classes
        if '{}' in prompt:
            texts = [prompt.replace('{}', name.replace("_", " "), 1)
                    for name in self.classnames]
        else:
            texts = [f"{prompt} {name.replace('_', ' ')}"
                    for name in self.classnames]

        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cache (limit cache size)
        if len(self._text_feature_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self._text_feature_cache))
            del self._text_feature_cache[oldest_key]
        self._text_feature_cache[prompt] = text_features

        return text_features

    def compute_detailed_metrics(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompt: str
    ) -> Dict[str, float]:
        """
        Compute detailed metrics for a single prompt on a batch.

        Returns:
            Dictionary with accuracy, margin, confidence, and per-sample info
        """
        with torch.no_grad():
            text_features = self._get_text_features(prompt)

            # Encode images
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            # Clamp logits to prevent numerical issues
            logits = torch.clamp(logits, min=-100, max=100)

            # Predictions
            preds = logits.argmax(dim=-1)
            correct_mask = (preds == labels).float()
            accuracy = correct_mask.mean().item()

            # Softmax probabilities with numerical stability
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-10, max=1.0)  # Prevent log(0)

            # Correct class probabilities
            correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()

            # Max wrong class probabilities
            mask = torch.ones_like(probs)
            mask.scatter_(1, labels.unsqueeze(1), 0)
            max_wrong_probs = (probs * mask).max(dim=-1)[0]

            # Margin: difference between correct and max wrong
            margins = correct_probs - max_wrong_probs
            mean_margin = margins.mean().item()

            # Confidence: entropy-based (lower entropy = higher confidence)
            # Add small epsilon to prevent log(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            max_entropy = np.log(len(self.classnames))
            normalized_confidence = 1.0 - (entropy.mean().item() / (max_entropy + 1e-10))

            # Clamp confidence to valid range
            normalized_confidence = np.clip(normalized_confidence, 0.0, 1.0)

            # Per-sample difficulty (for potential curriculum learning)
            sample_difficulties = 1.0 - correct_probs.cpu().numpy()

            # Final NaN check
            accuracy = 0.0 if np.isnan(accuracy) else accuracy
            mean_margin = 0.0 if np.isnan(mean_margin) else mean_margin
            normalized_confidence = 0.5 if np.isnan(normalized_confidence) else normalized_confidence

            return {
                'accuracy': accuracy,
                'margin': mean_margin,
                'confidence': normalized_confidence,
                'correct_probs': correct_probs.cpu().numpy(),
                'margins': margins.cpu().numpy(),
                'sample_difficulties': sample_difficulties,
                'num_samples': len(labels)
            }

    def compute_stable_reward_single_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompt: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a single batch with improved signal.

        Returns:
            Tuple of (reward, metrics_dict)
        """
        metrics = self.compute_detailed_metrics(images, labels, prompt)

        # Composite reward with multiple signals
        raw_reward = (
            self.acc_weight * metrics['accuracy'] +
            self.margin_weight * metrics['margin'] +
            self.confidence_weight * metrics['confidence']
        )

        metrics['raw_reward'] = raw_reward
        return raw_reward, metrics

    def compute_stable_reward_multi_batch(
        self,
        data_loader,
        prompt: str,
        num_batches: int = 3,
        return_std: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute stable reward by aggregating over multiple batches.
        This reduces variance from small batch sizes.

        Args:
            data_loader: DataLoader to sample batches from
            prompt: Text prompt to evaluate
            num_batches: Number of batches to aggregate over
            return_std: Whether to return standard deviation

        Returns:
            Tuple of (mean_reward, aggregated_metrics)
        """
        all_metrics = {
            'accuracy': [],
            'margin': [],
            'confidence': [],
            'raw_reward': []
        }
        total_samples = 0

        batch_iter = iter(data_loader)
        for _ in range(num_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                batch = next(batch_iter)

            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            reward, metrics = self.compute_stable_reward_single_batch(
                images, labels, prompt
            )

            # Weight by number of samples
            n = metrics['num_samples']
            total_samples += n

            for key in all_metrics:
                all_metrics[key].append(metrics[key] * n)

        # Compute weighted averages
        aggregated = {
            key: sum(values) / total_samples
            for key, values in all_metrics.items()
        }
        aggregated['num_samples'] = total_samples

        if return_std:
            # Compute standard deviation of rewards across batches
            reward_values = [m / (total_samples / num_batches)
                           for m in all_metrics['raw_reward']]
            aggregated['reward_std'] = np.std(reward_values)

        return aggregated['raw_reward'], aggregated

    def normalize_rewards(
        self,
        rewards: List[float],
        method: str = 'auto'
    ) -> List[float]:
        """
        Normalize a batch of rewards to reduce variance.

        Args:
            rewards: List of raw reward values
            method: 'zscore', 'rank', 'minmax', or 'auto'

        Returns:
            Normalized rewards
        """
        rewards = np.array(rewards)

        # Handle NaN/Inf values
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)

        if method == 'auto':
            if self.use_rank_normalization:
                method = 'rank'
            elif self.use_zscore_normalization:
                method = 'zscore'
            else:
                return rewards.tolist()

        if method == 'zscore':
            # Z-score normalization with running statistics
            mean = np.mean(rewards)
            std = np.std(rewards) + 1e-8
            normalized = (rewards - mean) / std

        elif method == 'rank':
            # Rank-based normalization (more robust to outliers)
            ranks = np.argsort(np.argsort(rewards)).astype(float)
            # Scale to [-1, 1]
            n = len(ranks)
            if n > 1:
                normalized = 2 * (ranks / (n - 1)) - 1
            else:
                normalized = np.zeros_like(rewards)

        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_r = np.min(rewards)
            max_r = np.max(rewards)
            range_r = max_r - min_r
            if range_r > 1e-8:
                normalized = (rewards - min_r) / range_r
            else:
                normalized = np.zeros_like(rewards)

        elif method == 'baseline':
            # Subtract running baseline
            normalized = rewards - self.running_mean

        else:
            normalized = rewards

        # Final clamp
        normalized = np.clip(normalized, -10.0, 10.0)

        return normalized.tolist()

    def compute_baseline_adjusted_reward(
        self,
        raw_reward: float
    ) -> float:
        """
        Compute reward adjusted by running baseline for variance reduction.
        This is a common technique in policy gradient methods.

        Args:
            raw_reward: The raw computed reward

        Returns:
            Baseline-adjusted reward (advantage)
        """
        # Handle NaN input
        if np.isnan(raw_reward) or np.isinf(raw_reward):
            raw_reward = 0.0

        # Update running statistics
        self.reward_history.append(raw_reward)

        if len(self.reward_history) >= 2:
            self.running_mean = np.mean(self.reward_history)
            self.running_std = np.std(self.reward_history) + 1e-8

        # Return advantage (reward - baseline)
        advantage = (raw_reward - self.running_mean) / self.running_std

        # Clamp advantage to prevent extreme values
        advantage = np.clip(advantage, -10.0, 10.0)

        return advantage

    def get_statistics(self) -> Dict[str, float]:
        """Get current running statistics."""
        return {
            'running_mean': self.running_mean,
            'running_std': self.running_std,
            'history_size': len(self.reward_history)
        }


# =============================================================================
# SOLUTION 2: Adaptive Reward Weighting
# =============================================================================

class AdaptiveRewardWeighter:
    """
    Dynamically adjusts reward component weights based on training progress
    and signal quality.
    """

    def __init__(
        self,
        initial_acc_weight: float = 30.0,
        initial_margin_weight: float = 10.0,
        initial_confidence_weight: float = 0.3,
        adaptation_rate: float = 0.01,
        target_reward_std: float = 5.0
    ):
        self.acc_weight = initial_acc_weight
        self.margin_weight = initial_margin_weight
        self.confidence_weight = initial_confidence_weight
        self.adaptation_rate = adaptation_rate
        self.target_reward_std = target_reward_std

        # Track component contributions
        self.acc_contributions = deque(maxlen=100)
        self.margin_contributions = deque(maxlen=100)
        self.confidence_contributions = deque(maxlen=100)
        self.reward_stds = deque(maxlen=100)

    def compute_reward(
        self,
        accuracy: float,
        margin: float,
        confidence: float
    ) -> float:
        """Compute weighted reward."""
        acc_contrib = self.acc_weight * accuracy
        margin_contrib = self.margin_weight * margin
        conf_contrib = self.confidence_weight * confidence

        # Track contributions
        self.acc_contributions.append(acc_contrib)
        self.margin_contributions.append(margin_contrib)
        self.confidence_contributions.append(conf_contrib)

        return acc_contrib + margin_contrib + conf_contrib

    def update_weights(self, recent_reward_std: float):
        """
        Adapt weights based on reward variance.
        If variance is too high, increase weight on more stable signals.
        """
        self.reward_stds.append(recent_reward_std)

        if len(self.reward_stds) < 10:
            return

        avg_std = np.mean(self.reward_stds)

        # If std is too high, shift weight to more stable signals
        if avg_std > self.target_reward_std * 1.5:
            # Margin and confidence are typically more stable than raw accuracy
            self.margin_weight *= (1 + self.adaptation_rate)
            self.confidence_weight *= (1 + self.adaptation_rate)
            self.acc_weight *= (1 - self.adaptation_rate * 0.5)

        elif avg_std < self.target_reward_std * 0.5:
            # Can afford more aggressive accuracy-based learning
            self.acc_weight *= (1 + self.adaptation_rate * 0.5)

        # Normalize to keep total scale similar
        total = self.acc_weight + self.margin_weight + self.confidence_weight
        scale = (30 + 10 + 0.3) / total
        self.acc_weight *= scale
        self.margin_weight *= scale
        self.confidence_weight *= scale

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return {
            'acc_weight': self.acc_weight,
            'margin_weight': self.margin_weight,
            'confidence_weight': self.confidence_weight
        }


# =============================================================================
# SOLUTION 3: Reward Smoother with Exponential Moving Average
# =============================================================================

class RewardSmoother:
    """
    Applies exponential moving average smoothing to rewards for each prompt,
    reducing noise from batch-to-batch variance.
    """

    def __init__(self, ema_alpha: float = 0.3, min_observations: int = 2):
        """
        Args:
            ema_alpha: Smoothing factor (higher = more weight on recent)
            min_observations: Minimum observations before using EMA
        """
        self.ema_alpha = ema_alpha
        self.min_observations = min_observations
        self.prompt_stats = {}  # prompt -> {'ema': float, 'count': int, 'history': list}

    def update_and_get_reward(
        self,
        prompt: str,
        raw_reward: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Update statistics for a prompt and return smoothed reward.

        Returns:
            Tuple of (smoothed_reward, stats_dict)
        """
        if prompt not in self.prompt_stats:
            self.prompt_stats[prompt] = {
                'ema': raw_reward,
                'count': 1,
                'history': [raw_reward],
                'raw_mean': raw_reward,
                'raw_std': 0.0
            }
            return raw_reward, self.prompt_stats[prompt]

        stats = self.prompt_stats[prompt]
        stats['count'] += 1
        stats['history'].append(raw_reward)

        # Keep history bounded
        if len(stats['history']) > 20:
            stats['history'] = stats['history'][-20:]

        # Update running statistics
        stats['raw_mean'] = np.mean(stats['history'])
        stats['raw_std'] = np.std(stats['history'])

        # EMA update
        if stats['count'] >= self.min_observations:
            stats['ema'] = self.ema_alpha * raw_reward + (1 - self.ema_alpha) * stats['ema']
        else:
            stats['ema'] = stats['raw_mean']

        return stats['ema'], stats

    def get_prompt_reliability(self, prompt: str) -> float:
        """
        Get reliability score for a prompt based on variance of its rewards.
        Lower variance = higher reliability.
        """
        if prompt not in self.prompt_stats:
            return 0.0

        stats = self.prompt_stats[prompt]
        if stats['count'] < 3:
            return 0.5  # Uncertain

        # Reliability inversely proportional to coefficient of variation
        cv = stats['raw_std'] / (abs(stats['raw_mean']) + 1e-8)
        reliability = 1.0 / (1.0 + cv)
        return reliability


# =============================================================================
# SOLUTION 4: Batch-Aggregated Evaluation Function
# =============================================================================

def evaluate_prompt_stable(
    clip_model,
    train_loader,
    prompt: str,
    classnames: List[str],
    device: torch.device,
    num_batches: int = 3,
    template_format: bool = True
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate a prompt with multi-batch aggregation for stable metrics.

    This replaces the original evaluate_clip_prompt for training to get
    more reliable reward signals.

    Args:
        clip_model: CLIP model
        train_loader: Training data loader
        prompt: Text prompt to evaluate
        classnames: List of class names
        device: Device to use
        num_batches: Number of batches to aggregate
        template_format: Whether prompt uses {} template

    Returns:
        Tuple of (accuracy, softmax_diff, detailed_metrics)
    """
    all_correct = 0
    all_total = 0
    all_margins = []
    all_confidences = []

    # Build text features once
    with torch.no_grad():
        if template_format and '{}' in prompt:
            texts = [prompt.replace('{}', name.replace("_", " "), 1)
                    for name in classnames]
        else:
            texts = [f"{prompt} {name.replace('_', ' ')}"
                    for name in classnames]

        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    batch_iter = iter(train_loader)
    batch_accuracies = []

    for _ in range(num_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)

        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            # Encode images
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logit_scale = clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()
            batch_acc = correct.mean().item()
            batch_accuracies.append(batch_acc)

            all_correct += correct.sum().item()
            all_total += labels.size(0)

            # Softmax and margins
            probs = F.softmax(logits, dim=-1)
            correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()

            mask = torch.ones_like(probs)
            mask.scatter_(1, labels.unsqueeze(1), 0)
            max_wrong_probs = (probs * mask).max(dim=-1)[0]

            margins = correct_probs - max_wrong_probs
            all_margins.extend(margins.cpu().numpy().tolist())

            # Confidence (entropy-based)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            max_entropy = np.log(len(classnames))
            batch_confidence = 1.0 - (entropy.mean().item() / max_entropy)
            all_confidences.append(batch_confidence)

    # Aggregate metrics
    accuracy = all_correct / all_total
    mean_margin = np.mean(all_margins)
    mean_confidence = np.mean(all_confidences)

    # Variance metrics for debugging
    acc_std = np.std(batch_accuracies)
    margin_std = np.std(all_margins)

    metrics = {
        'accuracy': accuracy,
        'margin': mean_margin,
        'confidence': mean_confidence,
        'acc_std': acc_std,
        'margin_std': margin_std,
        'num_samples': all_total,
        'num_batches': num_batches
    }

    return accuracy, mean_margin, metrics


def compute_robust_reward(
    accuracy: float,
    margin: float,
    confidence: float,
    acc_weight: float = 30.0,
    margin_weight: float = 10.0,
    confidence_weight: float = 5.0,
    use_log_transform: bool = False
) -> float:
    """
    Compute a robust reward signal combining multiple metrics.

    The combination of accuracy, margin, and confidence provides a
    more informative gradient signal than accuracy alone.

    Args:
        accuracy: Classification accuracy [0, 1]
        margin: Softmax margin (correct - max_wrong) [-1, 1]
        confidence: Entropy-based confidence [0, 1]
        acc_weight: Weight for accuracy term
        margin_weight: Weight for margin term
        confidence_weight: Weight for confidence term
        use_log_transform: Whether to use log transform for better gradients

    Returns:
        Composite reward value
    """
    if use_log_transform:
        # Log transform for better gradient flow at extremes
        eps = 1e-6
        acc_term = acc_weight * np.log(accuracy + eps)
        margin_term = margin_weight * margin  # Already can be negative
        conf_term = confidence_weight * np.log(confidence + eps)
    else:
        acc_term = acc_weight * accuracy
        margin_term = margin_weight * margin
        conf_term = confidence_weight * confidence

    return acc_term + margin_term + conf_term


# =============================================================================
# Original utility classes and functions (kept for compatibility)
# =============================================================================

class TopAccuracyTextsNoDuplicates:
    """
    Priority queue for storing top-performing prompts.
    Same as original BBPT implementation.
    """
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text, ep):
        if text in self.only_text:
            print('Already exists')
            return False
        else:
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (accuracy, len(text), text, ep))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
            elif accuracy > self.heap[0][0]:
                removed_text = heapq.heappop(self.heap)[2]
                if removed_text in self.text_map:
                    self.text_map.pop(removed_text)
                heapq.heappush(self.heap, (accuracy, len(text), text, ep))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
        return False

    def get_top_texts(self):
        return sorted(
            [(accuracy, text, ep) for accuracy, _, text, ep in self.heap],
            reverse=True
        )


class TopAccuracyTextsScore:
    """
    Extended priority queue with additional score tracking.
    Same as original BBPT implementation.
    """
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text, ep, score):
        if text in self.only_text:
            print('Already exists')
            return False
        else:
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (accuracy, len(text), text, ep, score))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
            elif accuracy > self.heap[0][0]:
                removed_text = heapq.heappop(self.heap)[2]
                if removed_text in self.text_map:
                    self.text_map.pop(removed_text)
                heapq.heappush(self.heap, (accuracy, len(text), text, ep, score))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
        return False

    def get_top_texts(self):
        return sorted(
            [(accuracy, text, ep, score) for accuracy, _, text, ep, score in self.heap],
            reverse=True
        )


def get_vlm_examples(train_data, classnames, shot=5):
    """
    Get random examples from training data for in-context learning.
    Adapted from original BBPT got_example function.
    """
    examples = ''
    indices = random.sample(range(len(train_data)), min(shot, len(train_data)))

    for idx in indices:
        item = train_data[idx]
        classname = item.classname
        examples += f'Image: [image of {classname}]\nLabel: {classname}\n\n'

    return examples


def get_vlm_example_items(train_data, shot=5):
    """
    Get random example items from training data.
    Returns the actual Datum objects for accessing images.
    """
    indices = random.sample(range(len(train_data)), min(shot, len(train_data)))
    return [train_data[idx] for idx in indices]


def get_vlm_examples_with_template(train_data, classnames, template, shot=5):
    """
    Get random examples with a specific template format.
    """
    examples = ''
    indices = random.sample(range(len(train_data)), min(shot, len(train_data)))

    for idx in indices:
        item = train_data[idx]
        classname = item.classname.replace("_", " ")
        prompt = template.format(classname)
        examples += f'Input: [image]\nOutput: {prompt}\n\n'

    return examples


def encode_text_with_clip(clip_model, texts, device):
    """
    Encode text prompts using CLIP's text encoder.
    """
    with torch.no_grad():
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def encode_images_with_clip(clip_model, images):
    """
    Encode images using CLIP's image encoder.
    """
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def evaluate_clip_prompt(
    clip_model,
    images,
    labels,
    prompt,
    classnames,
    device,
    template_format=True
):
    """
    Evaluate a single prompt on CLIP.
    Returns accuracy and softmax difference (for reward computation).

    [ORIGINAL FUNCTION - kept for compatibility]
    """
    with torch.no_grad():
        # Build text prompts for all classes
        if template_format and '{}' in prompt:
            texts = [prompt.replace('{}', name.replace("_", " "), 1) for name in classnames]
        else:
            texts = [f"{prompt} {name.replace('_', ' ')}" for name in classnames]

        # Encode text
        text_features = encode_text_with_clip(clip_model, texts, device)

        # Encode images
        image_features = encode_images_with_clip(clip_model, images)

        # Compute logits
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).float().sum()
        accuracy = correct.item() / labels.size(0)

        # Compute softmax difference (reward signal)
        probs = F.softmax(logits, dim=-1)
        correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()

        # Get max probability among wrong classes
        mask = torch.ones_like(probs)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        max_wrong_probs = (probs * mask).max(dim=-1)[0]

        softmax_diff = (correct_probs - max_wrong_probs).mean().item()

    return accuracy, softmax_diff


def evaluate_clip_on_loader(
    clip_model,
    data_loader,
    prompts,
    classnames,
    device,
    template_format=True
):
    """
    Evaluate multiple prompts on a data loader.
    Following the original BBPT evaluation style.
    """
    accuracies = []

    for prompt in prompts:
        correct = 0
        total = 0

        with torch.no_grad():
            # Build text prompts for all classes
            if template_format and '{}' in prompt:
                texts = [prompt.replace('{}', name.replace("_", " "), 1) for name in classnames]
            else:
                texts = [f"{prompt} {name.replace('_', ' ')}" for name in classnames]

            # Encode text once
            text_features = encode_text_with_clip(clip_model, texts, device)

            for batch in data_loader:
                images = batch['img'].to(device)
                labels = batch['label'].to(device)

                # Encode images
                image_features = encode_images_with_clip(clip_model, images)

                # Compute logits
                logit_scale = clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                # Compute predictions
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).float().sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)

    return accuracies


def evaluate_clip_on_loader_with_softmax(
    clip_model,
    data_loader,
    prompts,
    classnames,
    device
):
    """
    Evaluate prompts and return both accuracy and softmax difference.
    """
    accuracies = []
    softmax_diffs = []

    for prompt in prompts:
        correct = 0
        total = 0
        total_sd = 0

        with torch.no_grad():
            if '{}' in prompt:
                texts = [prompt.replace('{}', name.replace("_", " "), 1) for name in classnames]
            else:
                texts = [f"{prompt} {name.replace('_', ' ')}" for name in classnames]

            text_features = encode_text_with_clip(clip_model, texts, device)

            for batch in data_loader:
                images = batch['img'].to(device)
                labels = batch['label'].to(device)

                image_features = encode_images_with_clip(clip_model, images)

                logit_scale = clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).float().sum().item()
                total += labels.size(0)

                # Softmax difference
                probs = F.softmax(logits, dim=-1)
                correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()
                mask = torch.ones_like(probs)
                mask.scatter_(1, labels.unsqueeze(1), 0)
                max_wrong_probs = (probs * mask).max(dim=-1)[0]
                total_sd += (correct_probs - max_wrong_probs).sum().item()

        accuracies.append(correct / total if total > 0 else 0)
        softmax_diffs.append(total_sd / total if total > 0 else 0)

    return accuracies, softmax_diffs

def get_human_prompt_examples(ALL_HUMAN_PROMPT_EXAMPLES,num_examples: int = 5, sort_by_accuracy: bool = False) -> List[Dict]:
    """
    Get human-crafted prompt examples with their accuracy scores.
    """
    examples = ALL_HUMAN_PROMPT_EXAMPLES.copy()

    if sort_by_accuracy:
        examples = sorted(examples, key=lambda x: x['accuracy'], reverse=True)
        return examples[:num_examples]
    else:
        return random.sample(examples, min(num_examples, len(examples)))


def format_human_prompt_examples(
    examples: List[Dict],
    format_style: str = 'default'
) -> str:
    """
    Format human prompt examples as a string for in-context learning.
    """
    if format_style == 'detailed':
        formatted = "Here are some example prompt templates and their classification accuracy on the target dataset:\n\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"  Prompt: {ex['prompt']}\n"
            formatted += f"  Accuracy: {ex['accuracy']}%\n\n"
        formatted += "Notice that better prompts tend to achieve higher accuracy. "
        formatted += "Try to generate a \"Prompt\" that could achieve even higher accuracy.\n"

    elif format_style == 'simple':
        formatted = ""
        for ex in examples:
            formatted += f"Prompt: {ex['prompt']} Accuracy: {ex['accuracy']}\n"

    else:  # default
        formatted = "Example prompt templates with their accuracy scores:\n"
        for ex in examples:
            formatted += f"- Prompt: \"{ex['prompt']}\" → Accuracy: {ex['accuracy']}%\n"
        formatted += "\n"

    return formatted


def build_agent_query_with_examples(
    ALL_HUMAN_PROMPT_EXAMPLES,
    meta_prompt: str,
    human_examples: List[Dict] = None,
    num_human_examples: int = 5,
    human_example_style: str = 'default',
    use_top_examples: bool = True
) -> str:
    """
    Build complete query for the agent model with human-crafted prompt examples.
    """
    if human_examples is None:
        human_examples = get_human_prompt_examples(
            ALL_HUMAN_PROMPT_EXAMPLES,
            num_examples=num_human_examples,
            sort_by_accuracy=use_top_examples
        )

    human_examples_str = format_human_prompt_examples(
        human_examples,
        format_style=human_example_style
    )

    query = f"""{meta_prompt}

{human_examples_str}

Based on the above examples and the accuracy scores of different prompt templates, 
generate a new prompt template that uses {{}} as a placeholder for the class name.
The prompt should help achieve high classification accuracy.

"""
    return query


def build_agent_messages_with_examples(
    ALL_HUMAN_PROMPT_EXAMPLES,
    meta_prompt: str,
    human_examples: List[Dict] = None,
    num_human_examples: int = 5,
    human_example_style: str = 'default',
    use_top_examples: bool = True
) -> List[Dict]:
    """
    Build chat messages for the agent model with human-crafted examples.
    Suitable for chat-based models like Gemma-2B-IT.
    """
    query = build_agent_query_with_examples(
        ALL_HUMAN_PROMPT_EXAMPLES,
        meta_prompt=meta_prompt,
        human_examples=human_examples,
        num_human_examples=num_human_examples,
        human_example_style=human_example_style,
        use_top_examples=use_top_examples
    )

    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "The generated prompt is: "}
    ]

    return messages


def add_custom_prompt_example(prompt: str, accuracy: float):
    """
    Add a custom prompt example to the global list.
    Useful for adding dynamically discovered good prompts.
    """
    HUMAN_PROMPT_EXAMPLES.append({
        "prompt": prompt,
        "accuracy": accuracy
    })


def update_examples_from_queue(HUMAN_PROMPT_EXAMPLES,queue: TopAccuracyTextsNoDuplicates, top_k: int = 3):
    """
    Update the human examples with the best prompts discovered during training.
    This allows dynamic improvement of in-context examples.
    """
    top_prompts = queue.get_top_texts()[:top_k]

    for accuracy, prompt, _ in top_prompts:
        existing_prompts = [ex['prompt'] for ex in HUMAN_PROMPT_EXAMPLES]
        if prompt not in existing_prompts:
            HUMAN_PROMPT_EXAMPLES.append({
                "prompt": prompt,
                "accuracy": accuracy * 100
            })


# =============================================================================
# CONTROL VARIATE REWARD STRATEGY
# =============================================================================
#
# This implements a statistically optimal reward stabilization technique
# using a baseline prompt as a control variate to reduce variance.
#
# Key Formula:
#     R = r_π - α*(r_base - μ_base)
#
# Where:
#     - r_π: raw reward from generated prompt
#     - r_base: reward from baseline prompt on same batch
#     - μ_base: pre-computed global expected reward of baseline
#     - α*: optimal control coefficient = Cov(r_π, r_base) / Var(r_base)
#
# This transformation is unbiased (E[R] = E[r_π]) and achieves minimum variance
# among all linear baseline estimators.
# =============================================================================

from dataclasses import dataclass


@dataclass
class ControlVariateConfig:
    """Configuration for control variate reward strategy."""
    # Baseline prompt template
    baseline_prompt: str = "a photo of a {}."

    # EMA decay factor for online estimation
    ema_beta: float = 0.99

    # Small constant for numerical stability
    epsilon: float = 1e-8

    # Reward weights (same as original)
    acc_weight: float = 30.0
    softmax_diff_weight: float = 10.0

    # Minimum samples before using adaptive alpha
    warmup_steps: int = 20

    # Clamp alpha to reasonable range
    alpha_min: float = 0.0
    alpha_max: float = 2.0


class ControlVariateRewardComputer:
    """
    Computes stabilized rewards using control variate technique.

    The baseline prompt serves as an anchor to measure batch difficulty,
    and the optimal control coefficient α* is estimated online to achieve
    minimum variance in the reward signal.

    Mathematical Background:
    ------------------------
    R = r_π - α*(r_base - μ_base)

    The optimal α* that minimizes Var(R) is:
        α* = Cov(r_π, r_base) / Var(r_base)

    This is estimated online using EMA:
        σ²_base,t ← β * σ²_base,t-1 + (1-β) * (r_base - μ_base)²
        σ_cross,t ← β * σ_cross,t-1 + (1-β) * (r_π - r̄_π) * (r_base - μ_base)
        α*_t = σ_cross,t / (σ²_base,t + ε)
    """

    def __init__(
        self,
        clip_model,
        classnames: List[str],
        device: torch.device,
        config: Optional[ControlVariateConfig] = None
    ):
        """
        Args:
            clip_model: Frozen CLIP model for evaluation
            classnames: List of class names for the dataset
            device: Device for computation
            config: Configuration for control variate strategy
        """
        self.clip_model = clip_model
        self.classnames = classnames
        self.device = device
        self.config = config or ControlVariateConfig()

        # Pre-compute and cache baseline text features
        self._baseline_text_features = self._encode_baseline_prompt()

        # Global baseline statistics (to be computed before training)
        self.mu_base: float = 0.0  # E[r_base]
        self.is_calibrated: bool = False

        # Online EMA estimates
        self.var_base_ema: float = 1.0      # Var(r_base)
        self.cov_cross_ema: float = 0.5     # Cov(r_π, r_base)
        self.mean_r_pi_ema: float = 0.0     # Running mean of r_π

        # Step counter
        self.step_count: int = 0

        # History for debugging/analysis
        self.alpha_history: List[float] = []
        self.reward_history: List[Dict] = []

    def _encode_baseline_prompt(self) -> torch.Tensor:
        """Pre-compute text features for baseline prompt."""
        baseline_texts = [
            self.config.baseline_prompt.replace('{}', name.replace("_", " "), 1)
            for name in self.classnames
        ]

        with torch.no_grad():
            text_tokens = clip.tokenize(baseline_texts, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a prompt template into CLIP text features."""
        if '{}' in prompt:
            texts = [prompt.replace('{}', name.replace("_", " "), 1)
                    for name in self.classnames]
        else:
            texts = [f"{prompt} {name.replace('_', ' ')}"
                    for name in self.classnames]

        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _compute_reward_from_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute reward (accuracy + softmax_diff) given pre-encoded features.

        Returns:
            Tuple of (accuracy, softmax_diff)
        """
        with torch.no_grad():
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            # Accuracy
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()

            # Softmax difference
            probs = F.softmax(logits, dim=-1)
            correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze()

            # Max probability among wrong classes
            mask = torch.ones_like(probs)
            mask.scatter_(1, labels.unsqueeze(1), 0)
            max_wrong_probs = (probs * mask).max(dim=-1)[0]

            softmax_diff = (correct_probs - max_wrong_probs).mean().item()

        return accuracy, softmax_diff

    def _raw_reward(self, accuracy: float, softmax_diff: float) -> float:
        """Combine accuracy and softmax_diff into raw reward."""
        return (self.config.acc_weight * accuracy +
                self.config.softmax_diff_weight * softmax_diff)

    def calibrate_baseline(
        self,
        data_loader,
        num_batches: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Pre-compute global baseline statistics μ_base = E[r_base].

        This should be called ONCE before training starts.

        Args:
            data_loader: DataLoader for the training/calibration set
            num_batches: Number of batches to use (None = all)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with calibration statistics
        """
        print("\n" + "="*60)
        print("Calibrating Baseline Prompt Statistics")
        print("="*60)
        print(f"Baseline prompt: \"{self.config.baseline_prompt}\"")

        baseline_rewards = []
        baseline_accuracies = []
        baseline_softmax_diffs = []

        batch_iter = iter(data_loader)
        total_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

        iterator = tqdm(range(total_batches), desc="Calibrating") if show_progress else range(total_batches)

        for _ in iterator:
            try:
                batch = next(batch_iter)
            except StopIteration:
                break

            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            acc, sd = self._compute_reward_from_features(
                image_features, self._baseline_text_features, labels
            )

            reward = self._raw_reward(acc, sd)
            baseline_rewards.append(reward)
            baseline_accuracies.append(acc)
            baseline_softmax_diffs.append(sd)

        # Compute global statistics
        self.mu_base = np.mean(baseline_rewards)
        baseline_var = np.var(baseline_rewards)

        # Initialize EMA with empirical variance
        self.var_base_ema = baseline_var if baseline_var > 0 else 1.0

        self.is_calibrated = True

        stats = {
            'mu_base': self.mu_base,
            'var_base': baseline_var,
            'std_base': np.sqrt(baseline_var),
            'mean_accuracy': np.mean(baseline_accuracies),
            'mean_softmax_diff': np.mean(baseline_softmax_diffs),
            'num_batches': len(baseline_rewards)
        }

        print(f"\nCalibration Results:")
        print(f"  μ_base (global mean reward): {stats['mu_base']:.4f}")
        print(f"  σ_base (std of baseline):    {stats['std_base']:.4f}")
        print(f"  Mean accuracy:               {stats['mean_accuracy']:.4f}")
        print(f"  Mean softmax diff:           {stats['mean_softmax_diff']:.4f}")
        print("="*60 + "\n")

        return stats

    def compute_baseline_reward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute baseline reward for the current batch.

        Args:
            images: Batch of images
            labels: Batch of labels

        Returns:
            Tuple of (r_base, accuracy, softmax_diff)
        """
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        acc, sd = self._compute_reward_from_features(
            image_features, self._baseline_text_features, labels
        )

        r_base = self._raw_reward(acc, sd)
        return r_base, acc, sd

    def compute_prompt_reward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompt: str
    ) -> Tuple[float, float, float]:
        """
        Compute raw reward for a generated prompt.

        Args:
            images: Batch of images
            labels: Batch of labels
            prompt: Generated prompt template

        Returns:
            Tuple of (r_pi, accuracy, softmax_diff)
        """
        text_features = self._encode_prompt(prompt)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        acc, sd = self._compute_reward_from_features(
            image_features, text_features, labels
        )

        r_pi = self._raw_reward(acc, sd)
        return r_pi, acc, sd

    def _update_ema_statistics(
        self,
        r_pi: float,
        r_base: float
    ) -> None:
        """Update EMA estimates of variance and covariance."""
        beta = self.config.ema_beta

        # Bias term for current batch
        base_deviation = r_base - self.mu_base

        # Update Var(r_base) estimate
        self.var_base_ema = (beta * self.var_base_ema +
                            (1 - beta) * (base_deviation ** 2))

        # Update mean of r_pi for covariance calculation
        old_mean = self.mean_r_pi_ema
        self.mean_r_pi_ema = beta * self.mean_r_pi_ema + (1 - beta) * r_pi

        # Update Cov(r_pi, r_base) estimate
        pi_deviation = r_pi - old_mean
        self.cov_cross_ema = (beta * self.cov_cross_ema +
                             (1 - beta) * pi_deviation * base_deviation)

    def compute_optimal_alpha(self) -> float:
        """
        Compute the optimal control coefficient α*.

        α* = Cov(r_π, r_base) / Var(r_base)

        Returns:
            Optimal alpha value
        """
        if self.step_count < self.config.warmup_steps:
            # During warmup, use conservative alpha = 1 (standard baseline subtraction)
            return 1.0

        alpha = self.cov_cross_ema / (self.var_base_ema + self.config.epsilon)

        # Clamp to reasonable range
        alpha = np.clip(alpha, self.config.alpha_min, self.config.alpha_max)

        return alpha

    def compute_stabilized_reward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompt: str,
        return_details: bool = False
    ):
        """
        Compute the stabilized reward using control variate technique.

        R = r_π - α*(r_base - μ_base)

        Args:
            images: Batch of images
            labels: Batch of labels
            prompt: Generated prompt template
            return_details: Whether to return detailed statistics

        Returns:
            Stabilized reward (and optionally details dict)
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Baseline not calibrated! Call calibrate_baseline() before training."
            )

        # Compute raw rewards
        r_pi, acc_pi, sd_pi = self.compute_prompt_reward(images, labels, prompt)
        r_base, acc_base, sd_base = self.compute_baseline_reward(images, labels)

        # Update EMA statistics
        self._update_ema_statistics(r_pi, r_base)
        self.step_count += 1

        # Compute optimal alpha
        alpha = self.compute_optimal_alpha()

        # Compute stabilized reward
        batch_bias = r_base - self.mu_base
        R = r_pi - alpha * batch_bias

        # Track history
        self.alpha_history.append(alpha)

        if return_details:
            details = {
                'r_pi': r_pi,
                'r_base': r_base,
                'mu_base': self.mu_base,
                'batch_bias': batch_bias,
                'alpha': alpha,
                'stabilized_reward': R,
                'accuracy_pi': acc_pi,
                'accuracy_base': acc_base,
                'softmax_diff_pi': sd_pi,
                'softmax_diff_base': sd_base,
                'var_base_ema': self.var_base_ema,
                'cov_cross_ema': self.cov_cross_ema,
            }
            self.reward_history.append(details)
            return R, details

        return R

    def compute_stabilized_rewards_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompts: List[str]
    ) -> Tuple[List[float], List[float], Dict]:
        """
        Compute stabilized rewards for multiple prompts on the same batch.

        This is more efficient as baseline only needs to be computed once.

        Args:
            images: Batch of images
            labels: Batch of labels
            prompts: List of generated prompt templates

        Returns:
            Tuple of (list of stabilized rewards, list of accuracies, summary statistics)
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Baseline not calibrated! Call calibrate_baseline() before training."
            )

        # Compute baseline reward once for the batch
        r_base, acc_base, sd_base = self.compute_baseline_reward(images, labels)
        batch_bias = r_base - self.mu_base

        stabilized_rewards = []
        raw_rewards = []
        accuracies = []

        # Encode images once
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for prompt in prompts:
            text_features = self._encode_prompt(prompt)
            acc, sd = self._compute_reward_from_features(
                image_features, text_features, labels
            )
            r_pi = self._raw_reward(acc, sd)

            # Update EMA and get alpha
            self._update_ema_statistics(r_pi, r_base)
            self.step_count += 1
            alpha = self.compute_optimal_alpha()

            # Stabilized reward
            R = r_pi - alpha * batch_bias

            stabilized_rewards.append(R)
            raw_rewards.append(r_pi)
            accuracies.append(acc)
            self.alpha_history.append(alpha)

        summary = {
            'r_base': r_base,
            'batch_bias': batch_bias,
            'alpha': alpha,  # Final alpha
            'mean_raw_reward': np.mean(raw_rewards),
            'mean_stabilized_reward': np.mean(stabilized_rewards),
            'mean_accuracy': np.mean(accuracies),
            'accuracy_base': acc_base,
            'var_base_ema': self.var_base_ema,
            'cov_cross_ema': self.cov_cross_ema,
        }

        return stabilized_rewards, accuracies, summary

    def get_statistics(self) -> Dict:
        """Get current statistics for logging."""
        alpha = self.compute_optimal_alpha()

        return {
            'mu_base': self.mu_base,
            'var_base_ema': self.var_base_ema,
            'cov_cross_ema': self.cov_cross_ema,
            'mean_r_pi_ema': self.mean_r_pi_ema,
            'current_alpha': alpha,
            'step_count': self.step_count,
            'mean_alpha': np.mean(self.alpha_history) if self.alpha_history else 0.0,
            'std_alpha': np.std(self.alpha_history) if self.alpha_history else 0.0,
        }

    def reset_ema(self) -> None:
        """Reset EMA statistics (e.g., for a new training run)."""
        self.var_base_ema = 1.0
        self.cov_cross_ema = 0.5
        self.mean_r_pi_ema = 0.0
        self.step_count = 0
        self.alpha_history = []
        self.reward_history = []


def create_control_variate_reward_computer(
    clip_model,
    classnames: List[str],
    device: torch.device,
    baseline_prompt: str = "a photo of a {}.",
    ema_beta: float = 0.99,
    acc_weight: float = 30.0,
    softmax_diff_weight: float = 10.0,
    warmup_steps: int = 20
) -> ControlVariateRewardComputer:
    """
    Factory function to create a ControlVariateRewardComputer.

    Args:
        clip_model: Frozen CLIP model
        classnames: List of class names
        device: Computation device
        baseline_prompt: Template for baseline prompt
        ema_beta: EMA decay factor
        acc_weight: Weight for accuracy in reward
        softmax_diff_weight: Weight for softmax difference in reward
        warmup_steps: Steps before using adaptive alpha

    Returns:
        Configured ControlVariateRewardComputer instance
    """
    config = ControlVariateConfig(
        baseline_prompt=baseline_prompt,
        ema_beta=ema_beta,
        acc_weight=acc_weight,
        softmax_diff_weight=softmax_diff_weight,
        warmup_steps=warmup_steps
    )

    return ControlVariateRewardComputer(
        clip_model=clip_model,
        classnames=classnames,
        device=device,
        config=config
    )