#!/bin/bash

seeds=(7 14 21 28 35 42 49 56 63 70)

for seed in ${seeds[@]}; do
  echo "Running $1 with seed: $seed..."
  python -m entity_networks.main --dataset=$1 --seed=$seed
  sleep 10
done
