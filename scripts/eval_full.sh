#!/bin/bash

DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=`dirname "$0"`/../src

python $SRC_DIR/train_seq2seq.py \
    --do_predict \
    --eval_beams 5 \
    --predict_with_generate \
    --overwrite_output_dir \
    --tie_encoder_decoder \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 3 \
    --output_dir $OUT_DIR \
    --cache_dir /media/dmitry/data/huggingface/cache \
    --load_data_from /media/dmitry/data/outputs/headlines/ria_1k \
    --model_name_or_path /media/dmitry/other/resave_tpu_weights_v15_2ep
