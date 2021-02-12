#!/bin/bash

DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset/ria_1k.json
CACHE_DIR=/media/dmitry/data/huggingface/cache_1
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=$(dirname "$0")/../src

python "$SRC_DIR/run_bert_score.py" \
    --batch_size 256 \
    --output_dir $OUT_DIR \
    --cache_dir $CACHE_DIR \
    --test_data "$DATA_DIR" \
    --test_size 0.1 \
    --predictions "$OUT_DIR/test/test_generations_v39.txt"
