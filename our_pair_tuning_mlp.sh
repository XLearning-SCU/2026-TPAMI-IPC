#!/bin/bash


#datasets=("clevr4" "cards")
#criteria=(
#    "texture shape color count"
#    "number suits"
#)
#datasets=("fruit360" "cifar10")
#criteria=(
#    "color species"
#    "object scene"
#)
#datasets=("stanford-cars" "cub")
#criteria=(
#    "brand color type"
#    "species action scene"
#)
#datasets=("clevr4" "cards" "stanford-cars" "cub" "fruit360" "cifar10")
#criteria=(
#    "texture shape color count"
#    "number suits"
#    "brand color type"
#    "species action scene"
#    "color species"
#    "object scene"
#)


datasets=("gtsrb")
criteria=(
    "type"
)


pairs=(500)
each_num=5



for pair in "${pairs[@]}"; do
    for i in "${!datasets[@]}"; do
        criterion=(${criteria[$i]})  # 将字符串拆分成数组
        dataset=${datasets[$i]}
        for c in "${criterion[@]}"; do
          echo "Running command: python tuning_pair_mlp.py --criterion=$c --dataset=$dataset --each_num=$each_num --random=0"
          CUDA_VISIBLE_DEVICES=0 python tuning_pair_mlp.py --criterion=$c --dataset=$dataset --each_num=$each_num --random=0
        done
        echo "----------------------"
    done
done