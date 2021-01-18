#!/bin/bash

DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=`dirname "$0"`/../src

python $SRC_DIR/train_seq2seq.py \
    --do_train \
    --per_device_train_batch_size 3 \
    --logging_steps 1 \
    --data_json $DATA_DIR/ria_1k.json \
    --output_dir $OUT_DIR
