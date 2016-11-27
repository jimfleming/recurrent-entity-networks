#!/bin/bash

seeds=(7,13,17,23,27,31,37,43,47,53)

for seed in ${seeds[@]}; do
  echo "Running $1 with seed: $seed..."
  python -m entity_networks.main --dataset=$1 --seed=$seed
  sleep 10
done
