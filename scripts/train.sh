#!/bin/bash

DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=`dirname "$0"`/../src

python $SRC_DIR/train_seq2seq.py \
    --do_eval \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --max_steps 5 \
    --eval_steps 10 \
    --eval_accumulation_steps 3 \
    --data_json $DATA_DIR/ria_1k.json \
    --output_dir $OUT_DIR
