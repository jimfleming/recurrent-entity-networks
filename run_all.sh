#!/bin/bash

for file in datasets/processed/*.json; do
  python -m entity_networks.main --dataset=$file
done
