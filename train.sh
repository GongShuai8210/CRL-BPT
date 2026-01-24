#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export HTTP_PROXY=http://202.194.251.49:7897
export HTTPS_PROXY=http://202.194.251.49:7897

#DATASETS=(
#"caltech101"
#"dtd"
#"eurosat"
#"fgvc_aircraft"
#"food101"
#"oxford_flowers"
#"oxford_pets"
#"resisc45"
#"standford_cars"
#"sun397"
#"svhn"
#"ucf101"
#"imagenet"
#)
DATASETS=(
"oxford_flowers"
)

PROMPT_LENGTHS=(
  15
)

# Common parameters
NUM_HUMAN_EXAMPLES=10
EPOCHS=1000
MAX_NUM_APIS=2000
NUM_SHOTS=16
BASELINE_PROMPT="a photo of a {}."
CV_EMA_BETA=0.99
CV_WARMUP_STEPS=5
CV_CALIBRATION_BATCHES=200

for DATASET in "${DATASETS[@]}"; do
    for PROMPT_LENGTH in "${PROMPT_LENGTHS[@]}"; do
        echo "========================================" | tee -a ${LOG_FILE}
        echo "PROMPT_LENGTH: ${PROMPT_LENGTH}" | tee -a ${LOG_FILE}
        echo "DATASET: ${DATASET}" | tee -a ${LOG_FILE}
        echo "========================================" | tee -a ${LOG_FILE}

        python stable_bbpt_curri_1.py \
            --clip_backbone ViT-B/16 \
            --batch_size 256 \
            --max_prompt_length ${PROMPT_LENGTH} \
            --dataset ${DATASET} \
            --num_human_examples ${NUM_HUMAN_EXAMPLES} \
            --prompt_per_image 4 \
            --epochs ${EPOCHS} \
            --max_num_apis ${MAX_NUM_APIS} \
            --num_shots ${NUM_SHOTS} \
            --baseline_prompt "${BASELINE_PROMPT}" \
            --cv_ema_beta ${CV_EMA_BETA} \
            --cv_warmup_steps ${CV_WARMUP_STEPS} \
            --cv_calibration_batches ${CV_CALIBRATION_BATCHES}
    done
done