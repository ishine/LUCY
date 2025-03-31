#!/bin/bash
WORK_DIR=$(pwd)
CACHE_DIR=/mnt/data/hetinggao/models

SAVE_AUDIO=True
TEXT_ONLY=False

MODEL_NAME_OR_PATH=/path/to/checkpoint
AUDIO_ENCODER=models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning

EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=demo
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX

mkdir -p $OUTPUT_PATH

export PYTHONPATH=$WORK_DIR
python vita/scripts/demo.py \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
    --model_hidden_size 1536 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --text_vocab_size 152064 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "F10" "M29" "ER" \
    --max_code_length 1000 \
    --max_keep_sample_size $((25*16000)) \
    --output_path ${OUTPUT_PATH} \
    --save_audio ${SAVE_AUDIO} \
    --output_text_only ${TEXT_ONLY} \
