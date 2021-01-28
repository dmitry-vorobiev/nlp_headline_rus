#!/bin/bash

BASE_DIR=$HOME/projects/2gis_sum
DATA_DIR=$BASE_DIR/ria_news_dataset
WEIGHTS_DIR=$BASE_DIR/rubert-base-cased
PREP_DATA_DIR=/media/dmitry/data/outputs/headlines/ria_1k
CACHE_DIR=/media/dmitry/data/huggingface/cache
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=`dirname "$0"`/../src

python $SRC_DIR/train_seq2seq.py \
    --do_train \
    --fp16 \
    --predict_with_generate \
    --overwrite_output_dir \
    --tie_encoder_decoder \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --learning_rate 2e-4 \
    --max_steps -1 \
    --save_steps 500 \
    --eval_accumulation_steps 3 \
    --output_dir $OUT_DIR \
    --cache_dir $CACHE_DIR \
    --load_data_from $PREP_DATA_DIR \
    --model_name_or_path $WEIGHTS_DIR
