#!/bin/bash

# DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset
WEIGHTS_DIR=/media/dmitry/other/resave_tpu_weights_v23_3ep
PREP_DATA_DIR=/media/dmitry/other/2021-02-01_preprocessed_ria_small_test
CACHE_DIR=/media/dmitry/data/huggingface/cache
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=$(dirname "$0")/../src

python -m torch.distributed.launch --nproc_per_node 2 "$SRC_DIR/train_seq2seq.py" \
    --do_predict \
    --eval_beams 5 \
    --predict_with_generate \
    --overwrite_output_dir \
    --tie_encoder_decoder \
    --per_device_eval_batch_size 12 \
    --eval_accumulation_steps 2 \
    --output_dir $OUT_DIR \
    --cache_dir $CACHE_DIR \
    --load_data_from $PREP_DATA_DIR \
    --model_name_or_path "$WEIGHTS_DIR"
