#!/bin/bash

DATA_DIR=$HOME/projects/2gis_sum/ria_news_dataset/ria_1k.json
CACHE_DIR=/media/dmitry/data/huggingface/cache_1
OUT_DIR=/media/dmitry/data/outputs/headlines/default
SRC_DIR=$(dirname "$0")/../src

python "$SRC_DIR/run_bert_score.py" \
    --output_dir $OUT_DIR \
    --cache_dir $CACHE_DIR \
    --data_json "$DATA_DIR" \
    --predictions_txt "$OUT_DIR/test/test_generations_v23.txt"
