#!/bin/bash

DATA_DIR=/io/ria_news_dataset
WEIGHTS_DIR=/io/rubert_ria_headlines
OUT_DIR=/io/output
SRC_DIR=`dirname "$0"`/../src

python $SRC_DIR/train_seq2seq.py \
    --do_predict \
    --eval_beams 5 \
    --predict_with_generate \
    --overwrite_output_dir \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 3 \
    --output_dir $OUT_DIR \
    --data_json $DATA_DIR/ria_1k.json \
    --model_name_or_path $WEIGHTS_DIR
