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
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --learning_rate 1e-4 \
    --max_steps 10 \
    --save_steps 500 \
    --eval_accumulation_steps 3 \
    --output_dir $OUT_DIR \
    --cache_dir /media/dmitry/data/huggingface/cache \
    --load_data_from /media/dmitry/data/outputs/headlines/ria_1k \
    --model_name_or_path /media/dmitry/data/outputs/headlines/2021-01-26_0.8_ep_gpu
    # --model_name_or_path /home/dmitry/projects/2gis_sum/rubert-base-cased
    # --model_name_or_path /media/dmitry/data/outputs/headlines/2021-01-25_train_seq2seq_ria_tpu/output
    # --model_name_or_path /media/dmitry/data/outputs/headlines/2021-01-23_cp_0.5_shared_tpu
