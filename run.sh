#!/bin/bash


K=2
Top_k=100
# llama2 zephyr vicuna
MODEL_NAME="zephyr"
# bm25 splade
FIRST_STAGE_RESULTS="bm25"
# 19 20
YEAR="20"
# explanation ablation
OTHER_PROMPT=""
# true 是随机 false是local
IS_RANDOM=false




COMMAND="python -m evaluation.reranker_eval --k $K --model_name $MODEL_NAME --year $YEAR --topk $Top_k"


if [ -n "$FIRST_STAGE_RESULTS" ]; then
    COMMAND="$COMMAND --first_stage_results $FIRST_STAGE_RESULTS"
fi

if [ -n "$OTHER_PROMPT" ]; then
    COMMAND="$COMMAND --other_prompt $OTHER_PROMPT"
fi

if $IS_RANDOM; then
    COMMAND="$COMMAND --is_random"
fi

echo $COMMAND
eval $COMMAND
