#!/bin/bash


seeds=(0 1 2 3 4)

datasets=("clevr4")
criteria=(
   "texture shape color"
)
#datasets=("clevr4" "cards")
#criteria=(
#    "count"
#    "number suits"
#)
# datasets=("cifar10")
# criteria=(
#     "object scene"
# )
#datasets=("stanford-cars")
#criteria=(
#    "brand color type"
#)
#datasets=("fruit360")
#criteria=(
#    "color species"
#)
#datasets=("cub")
#criteria=(
#    "species action scene"
#)



pairs=(500)

LOG_DIR=logs

for pair in "${pairs[@]}"; do
  for i in "${!datasets[@]}"; do
      criterion=(${criteria[$i]})  # 将字符串拆分成数组
      dataset=${datasets[$i]}
      for c in "${criterion[@]}"; do
          for seed in "${seeds[@]}"; do
            python random_pair.py --criterion=$c --dataset=$dataset --seed=$seed --pair_budget=$pair --random=1
            echo "Running command: python tuning_pair_mlp.py --criterion=$c --dataset=$dataset --seed=$seed --pair_budget=$pair --random=1"
            LOGFILE=$LOG_DIR/${c}_${pair}_${seed}.log
            CUDA_VISIBLE_DEVICES=0 python tuning_pair_mlp.py --criterion=$c --dataset=$dataset --seed=$seed --pair_budget=$pair --random=1 2>&1 | tee $LOGFILE
          done
      done
      echo "----------------------"
  done
done

