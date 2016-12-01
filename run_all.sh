#!/bin/bash

base_path="datasets/processed"

filenames=(
  # "$base_path/qa1_single-supporting-fact_10k.json"
  # LONG: "$base_path/qa2_two-supporting-facts_10k.json"
  # LONG: "$base_path/qa3_three-supporting-facts_10k.json"
  # "$base_path/qa4_two-arg-relations_10k.json"
  # LONG: "$base_path/qa5_three-arg-relations_10k.json"
  # "$base_path/qa6_yes-no-questions_10k.json"
  # LONG: "$base_path/qa7_counting_10k.json"
  # LONG: "$base_path/qa8_lists-sets_10k.json"
  # "$base_path/qa9_simple-negation_10k.json"
  # "$base_path/qa10_indefinite-knowledge_10k.json"
  # "$base_path/qa11_basic-coreference_10k.json"
  # "$base_path/qa12_conjunction_10k.json"
  # "$base_path/qa13_compound-coreference_10k.json"
  # "$base_path/qa14_time-reasoning_10k.json"
  # "$base_path/qa15_basic-deduction_10k.json"
  "$base_path/qa16_basic-induction_10k.json"
  # "$base_path/qa17_positional-reasoning_10k.json"
  # "$base_path/qa18_size-reasoning_10k.json"
  # FAIL: "$base_path/qa19_path-finding_10k.json"
  # "$base_path/qa20_agents-motivations_10k.json"
)

seeds=(7 14 21 28 35 42 49 56 63 70)

for file in ${filenames[@]}; do
  for seed in ${seeds[@]}; do
    echo "Running $file with seed $seed..."
    python -m entity_networks.main --dataset=$file --seed=$seed
    sleep 10
  done
done
