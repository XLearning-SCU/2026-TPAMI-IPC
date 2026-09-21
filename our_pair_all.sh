#!/bin/bash






datasets=("clevr4" "cards" "fruit360")
criteria=(
    "texture shape color count"
    "number suits"
    "color species"
)
#datasets=("cifar10")
#criteria=(
#    "object scene"
#)
#datasets=("stanford-cars")
#criteria=(
#    "brand color type"
#)
#datasets=("cub")
#criteria=(
#    "species action scene"
#)




each_num=5

pairs=(500 500 50)


for pair in "${pairs[@]}"; do
#  for point in "${points[@]}"; do
    for i in "${!datasets[@]}"; do
        criterion=(${criteria[$i]})  # 将字符串拆分成数组
        dataset=${datasets[$i]}
        for c in "${criterion[@]}"; do
            echo "Running command: python our_pair_all.py --criterion=$c --dataset=$dataset --pair_budget=$pair --each_num=$each_num --random=0"
            CUDA_VISIBLE_DEVICES=0 python our_pair_all.py --criterion=$c --dataset=$dataset --pair_budget=$pair --each_num=$each_num --random=0
          done
        done
        echo "----------------------"
    done
#  done
done



