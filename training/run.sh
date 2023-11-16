CUDA_VISIBLE_DEVICES=$2 nohup python $1 --task Metal --train --seed 2021 --run_id $3 > $3.log &
#CUDA_VISIBLE_DEVICES=$2 python $1 --task Metal --test --seed 2021 --run_id $3
