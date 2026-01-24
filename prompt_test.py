import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb2
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

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

import clip
import ast

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
# =============================================================================
# Unified PPO + Curriculum Trainer
# =============================================================================

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/16')
    parser.add_argument('--dataset_config_file', type=str, default='datasets_config')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--cache_dir', type=str, default='LLMs/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_prompt_length', type=int, default=15)
    parser.add_argument('--num_shots', type=int, default=16)
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
    parser.add_argument('--top_k', type=int, default=1)
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
    # cfg.DATASET.NAME = args.dataset
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
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
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

    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Setup ---
    print("Setting up DassL data manager...")
    cfg = setup_dassl_cfg(args)
    # for config_dataset_ in ['datasets_config/caltech101.yaml','datasets_config/dtd.yaml','datasets_config/oxford_pets.yaml',
    #                         'datasets_config/oxford_flowers.yaml','datasets_config/food101.yaml','datasets_config/ucf101.yaml',
    #                         'datasets_config/sun397.yaml','datasets_config/eurosat.yaml','datasets_config/fgvc_aircraft.yaml']:
    # for config_dataset_ in ['datasets_config/eurosat.yaml']:
    #     args.dataset_config_file = config_dataset_
    #     cfg.merge_from_file(args.dataset_config_file)
    #     dm = DataManager(cfg)
    #     classnames = dm.dataset.classnames
    #     test_loader = dm.test_loader
    #
    #     print(f"Loading CLIP (backbone: {args.clip_backbone})...")
    #     clip_model, clip_preprocess = load_clip_model(args.clip_backbone, device)
    #     print("Starting test....")
    #     test_acc = utils_r.evaluate_clip_on_loader(
    #         clip_model, test_loader, [prompt], classnames, device
    #     )
    #
    #     print(f"Test Acc: {test_acc[0]:.4f} (Prompt: {prompt})")
    #     print("="*25)

    config_dataset_ = 'datasets_config/eurosat.yaml'
    prompt="a photo of a hard to see {}."

    args.dataset_config_file = config_dataset_
    cfg.merge_from_file(args.dataset_config_file)
    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    test_loader = dm.test_loader

    print(f"Loading CLIP (backbone: {args.clip_backbone})...")
    clip_model, clip_preprocess = load_clip_model(args.clip_backbone, device)
    print("Starting test....")
    print(prompt)
    test_acc = utils_r.evaluate_clip_on_loader(
        clip_model, test_loader, [prompt], classnames, device
    )

    print(f"Test Acc: {test_acc[0]:.4f} (Prompt: {prompt})")
    print("="*25)

if __name__ == '__main__':
    main()

