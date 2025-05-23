#!/bin/bash
DATASET_DIR=
RESULT_DIR=.
MODEL_ARCH=vit-b16  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}.csv

python ./evaluate_CLIP.py \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/results/glaucoma_CLIP_${MODEL_ARCH} \
		--lr ${LR} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--pretrained_weights 'results/glaucoma_FairCLIP_vit-b16_race_seed1032_auc0.6428/clip_ep001.pth'
