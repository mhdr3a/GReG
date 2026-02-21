#!/bin/bash

MODEL="gpt-4o"
API_KEY=""
USING_BATCH_API=1  # set to 0 if you want to disable batch mode and skip caching batch_ids

# Dataset details
declare -A DATA_PATHS
DATA_PATHS["HotpotQA"]="data/HotpotQA/hotpot_dev_distractor_v1_sampled.jsonl"
DATA_PATHS["2WikiMultihopQA"]="data/2WikiMultihopQA/dev_sampled.jsonl"
DATA_PATHS["MuSiQue"]="data/MuSiQue/musique_ans_v1.0_dev_sampled.jsonl"

DATASETS=("HotpotQA" "2WikiMultihopQA" "MuSiQue")

mkdir -p cache

if [[ "$USING_BATCH_API" -eq 1 ]]; then
  for DATASET in "${DATASETS[@]}"; do
    DATA="${DATA_PATHS[$DATASET]}"

    echo "Running for $DATASET..."

    echo "Temperature = 0"
    # Temperature = 0, single run
    OUTPUT=$(echo "y" | python3 hint_generator.py \
      --data "$DATA" \
      --model "$MODEL" \
      --api_key "$API_KEY" \
      --dataset "$DATASET" \
      --using_batch_api 1)

    BATCH_ID=$(echo "$OUTPUT" | grep -oP 'batch_id:\s*\K\S+')
    echo "$BATCH_ID" > "cache/${DATASET}_0.txt"

    # Temperature = 0.7, k_fold = 1 to 4
    for KFOLD in {1..4}; do
      echo "Temperature = 0.7, k_fold = $KFOLD"
      OUTPUT=$(echo "y" | python3 hint_generator.py \
        --data "$DATA" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --dataset "$DATASET" \
        --temperature 0.7 \
        --k_fold "$KFOLD" \
        --using_batch_api 1)

      BATCH_ID=$(echo "$OUTPUT" | grep -oP 'batch_id:\s*\K\S+')
      echo "$BATCH_ID" > "cache/${DATASET}_0.7_${KFOLD}.txt"
    done
  done
else
  for DATASET in "${DATASETS[@]}"; do
    DATA="${DATA_PATHS[$DATASET]}"

    echo "Running for $DATASET..."

    echo "Temperature = 0"

    # Temperature = 0, single run
    echo "y" | python3 hint_generator.py \
      --data "$DATA" \
      --model "$MODEL" \
      --api_key "$API_KEY" \
      --dataset "$DATASET" \
      --using_batch_api 0

    # Temperature = 0.7, k_fold = 1 to 4
    for KFOLD in {1..4}; do
      echo "Temperature = 0.7, k_fold = $KFOLD"
      echo "y" | python3 hint_generator.py \
        --data "$DATA" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --dataset "$DATASET" \
        --temperature 0.7 \
        --k_fold "$KFOLD" \
        --using_batch_api 0
    done
  done
fi
