#!/bin/bash

download() {
  if [ -f $1 ]; then
    echo "$1 already exists"
  else
    echo "Downloading $1"
    wget $2 -O $1
  fi
}

# Download BurstGPT
mkdir -p burstgpt
download burstgpt/BurstGPT_without_fails.csv https://raw.githubusercontent.com/HPMLL/BurstGPT/80574fd1d9ed2647589fc84084e49b96df009d99/data/BurstGPT_without_fails.csv
# Download ShareGPT
mkdir -p sharegpt
download sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Download LongBench
if [ -d "longbench/raw" ]; then
  echo "longbench/raw already exists"
  exit 0
fi
pushd /tmp
rm -rf /tmp/data
download longbench.zip "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip?download=true"
unzip longbench.zip
popd
mkdir -p longbench
mv /tmp/data longbench/raw
