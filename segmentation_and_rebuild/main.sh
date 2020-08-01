#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --output=seg.txt
#SBATCH --mem=20G 

#   ---------SBATCH --time=0-10:00   

module load anaconda3/4.7.12
source activate py36

path_to_data_="../Data/forest4D_dlr/"
path_to_output_="../Data/forest4D_dlr/"

areas="testing/"
areas="training/"
areas="training/,testing/"

n_classes=4
reg_strength=.5
voxel=.01


python -u partition/segmentation.py --areas=$areas --n_labels=$n_classes --path_to_data=$path_to_data_ --path_to_output=$path_to_output_ --version="V0" --reg_strength=$reg_strength  --voxel_width=$voxel

#python -u partition/write_segments_to_pc.py --areas=$areas --db_test_name=$areas --n_classes=$n_classes --metrics=False --path_to_data=$path_to_data_ --path_to_output=$path_to_output_


