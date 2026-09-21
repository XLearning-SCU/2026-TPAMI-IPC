#!/bin/bash

datasets=("clevr4" "cards" "fruit360")
criteria=(
   "texture shape color count"
   "number suits"
   "color species"
)

pairs=(500 500 50)

seeds=(0 1 2 3 4)

noise_ratio=0.1

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    criterion_group="${criteria[$i]}"
    pair_budget="${pairs[$i]}"

    for criterion in $criterion_group; do
        for seed in "${seeds[@]}"; do
            echo "Running: dataset=$dataset, criterion=$criterion, seed=$seed, noise_ratio=$noise_ratio, pair_budget=$pair_budget"
            python ../tuning_pair_mlp.py \
                --dataset "$dataset" \
                --criterion "$criterion" \
                --pair_budget "$pair_budget" \
                --noise_ratio "$noise_ratio" \
                --seed "$seed"
        done
    done
done



