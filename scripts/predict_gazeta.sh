#!/bin/bash

DATA_JSON=$HOME/projects/2gis_sum/gazeta/gazeta_test_stripped.jsonl
WEIGHTS_DIR=/media/dmitry/other/resave_tpu_weights_v44_3ep
CACHE_DIR=/media/dmitry/data/huggingface/cache_1
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=$(dirname "$0")/../src

python "$SRC_DIR/train_seq2seq.py" \
    --do_predict \
    --predict_with_generate \
    --overwrite_output_dir \
    --eval_beams 5 \
    --eval_split 0.999 \
    --skip_text_clean \
    --tie_encoder_decoder \
    --max_source_length 512 \
    --max_target_length 48 \
    --val_max_target_length 48 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 1 \
    --output_dir $OUT_DIR \
    --cache_dir $CACHE_DIR \
    --data_json "$DATA_JSON" \
    --model_name_or_path "$WEIGHTS_DIR"
