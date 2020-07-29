#!/bin/bash
#SBATCH --job-name=forest
#SBATCH --output=forest.txt
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=40G     
#SBATCH --time=1-20:00   

module load nvidia/cuda/10.1.243
module load anaconda3/4.7.12

source activate tf

###################################################################################################################
dataset="forest4D_dlr"
path_in="../Data/forest4D_dlr/"

model_name="dlr_forest4d"
pre_train="dlr_forest4d"


n_classes=4
db_test_name="testing"
db_train_name="training"

num_gpus=1

python -u gpu_main.py --SEMA3D_PATH=$path_in --dataset=$dataset --num_gpus=$num_gpus --pre_train=$pre_train --model_name=$model_name --n_classes=$n_classes --db_test_name=$db_test_name --resume=1 --db_train_name=$db_train_name --ptn_npts=512 --training=True


