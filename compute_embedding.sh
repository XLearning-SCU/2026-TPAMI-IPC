#!/bin/bash



dataset=("clevr4" "cards" "cifar10" "stanford-cars" "fruit360" "cub")
criteria_list=(
   "texture shape color count"
   "number suits"
   "object scene"
   "brand color type"
   "color species"
   "species action scene"
)


# dataset=("gtsrb")
# criteria_list=(
#     "type"
# )
# dataset=("eurosat")
# criteria_list=(
#     "scene"
#     "human"
# )


# VLM="ALIGN"
# VLM="MetaCLIP"
#VLM="BLIP2"


for i in "${!dataset[@]}"; do
    criteria=(${criteria_list[$i]})  # 将字符串拆分成数组
    this_dataset=${dataset[$i]}
    echo "Dataset: $this_dataset"
    for c in "${criteria[@]}"; do
        echo "  Criteria: $c"
        echo "Running command: python image_embedding.py --criterion=$c --dataset=$this_dataset"
        python image_embedding.py --criterion=$c --dataset=$this_dataset
        echo "Running command: python text_embedding.py --criterion=$c --dataset=$this_dataset"
        python text_embedding.py --criterion=$c --dataset=$this_dataset
        echo "Running command: python sim_embedding.py --criterion=$c --dataset=$this_dataset"
        python sim_embedding.py --criterion=$c --dataset=$this_dataset
    done
    echo "----------------------"
done


