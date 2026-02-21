#!/bin/bash

MODEL="gpt-4o"
API_KEY=""

# Dataset details
declare -A DATA_PATHS
DATA_PATHS["HotpotQA"]="data/HotpotQA/hotpot_dev_distractor_v1_sampled.jsonl"
DATA_PATHS["2WikiMultihopQA"]="data/2WikiMultihopQA/dev_sampled.jsonl"
DATA_PATHS["MuSiQue"]="data/MuSiQue/musique_ans_v1.0_dev_sampled.jsonl"

DATASETS=("HotpotQA" "2WikiMultihopQA" "MuSiQue")

for DATASET in "${DATASETS[@]}"; do
  DATA="${DATA_PATHS[$DATASET]}"
  echo "Running for $DATASET..."
  echo "Temperature = 0"
  # Temperature = 0, single run
  BATCH_ID_FILE="cache/${DATASET}_0.txt"
  if [[ -f "$BATCH_ID_FILE" ]]; then
    BATCH_ID=$(cat "$BATCH_ID_FILE")
    python3 hint_generator.py \
      --data "$DATA" \
      --model "$MODEL" \
      --api_key "$API_KEY" \
      --dataset "$DATASET" \
      --using_prev_file 1 \
      --batch_id "$BATCH_ID"
  fi

  # Temperature = 0.7, k_fold = 1 to 4
  for KFOLD in {1..4}; do
    echo "Temperature = 0.7, k_fold = $KFOLD"
    BATCH_ID_FILE="cache/${DATASET}_0.7_${KFOLD}.txt"
    if [[ -f "$BATCH_ID_FILE" ]]; then
      BATCH_ID=$(cat "$BATCH_ID_FILE")
      python3 hint_generator.py \
        --data "$DATA" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --dataset "$DATASET" \
        --temperature 0.7 \
        --k_fold "$KFOLD" \
        --using_prev_file 1 \
        --batch_id "$BATCH_ID"
    fi
  done
done
