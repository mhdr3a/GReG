#!/bin/bash

BASE_DIR="data"

# MuSiQue
ZIP_NAME="musique_v1.0.zip"

# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME
mkdir -p "$BASE_DIR/MuSiQue/"
mv "$BASE_DIR"/*.json "$BASE_DIR/MuSiQue/"
mv "$BASE_DIR"/*.jsonl "$BASE_DIR/MuSiQue/"

# TODO: prevent these from zipping in.
rm -rf __MACOSX

# HotpotQA, IIRC, and 2WikiMultihopQA
download_and_unzip() {
  url=$1
  output_dir="$BASE_DIR/$2"
  new_filename=$3

  mkdir -p "$output_dir"

  echo "Downloading to $new_filename..."
  curl -L -o "$output_dir/$new_filename" "$url"

  case "$new_filename" in
    *.zip)
      echo "Unzipping $new_filename..."
      unzip -o "$output_dir/$new_filename" -d "$output_dir"
      rm "$output_dir/$new_filename"
      ;;
    *.tgz | *.tar.gz)
      echo "Extracting $new_filename..."
      tar -xzf "$output_dir/$new_filename" -C "$output_dir"
      rm "$output_dir/$new_filename"
      ;;
    *.json | *.jsonl)
      echo "Downloaded $new_filename"
      ;;
    *)
      echo "Unknown file extension for $new_filename"
      ;;
  esac
}

HOTPOTQA_TRAIN_URL="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
HOTPOTQA_DEV_URL="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
IIRC_URL="https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz"
TWIKIMULTIHOPQA_URL="https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1&dl=1"

download_and_unzip "$HOTPOTQA_TRAIN_URL" "HotpotQA" "hotpot_train_v1.1.json"
download_and_unzip "$HOTPOTQA_DEV_URL" "HotpotQA" "hotpot_dev_distractor_v1.json"
download_and_unzip "$IIRC_URL" "IIRC" "iirc_train_dev.tgz"
mv "$BASE_DIR"/IIRC/iirc_train_dev/*.json "$BASE_DIR"/IIRC/
rm -r "$BASE_DIR"/IIRC/iirc_train_dev
download_and_unzip "$TWIKIMULTIHOPQA_URL" "2WikiMultihopQA" "2WikiMultihopQA.zip"

echo "All datasets have been downloaded and processed."
