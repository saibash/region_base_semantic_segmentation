# **Welcome!**

This document will help you to install all the requirements for the project Region-Based Edge Convolutions With Geometric Attributes for the Semantic Segmentation of Large-Scale 3-D Point Clouds, link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9103287

author  : Jhonatan Contreras


System requirements.
- Anaconda &copy;
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
- In kratos server load module: module load anaconda3/4.7.12
- Clone the repository: https://gitlab.dlr.de/dw-bws/outdoor_semantic_segmentation

### Create py36 enviroment:
```bash
conda create --name py36_build --file spec-file_py36_build.txt
```
```bash
source activate py36_build
pip install git+https://github.com/pytorch/tnt.git@master
pip install -r requirements_py36_build.txt
```
```bash
conda create --name py36 --file spec-file_py36.txt
```
```bash
source activate py36
pip install git+https://github.com/pytorch/tnt.git@master
pip install -r requirements_py36.txt
```

## Compile source
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION

bash build.sh $CONDAENV

For intance :  bash build.sh "/home/cont_jh/anaconda3/envs/py36_build"

### Create tensorflow enviroment:
```bash
conda create --name tf --file spec-file_tf.txt
```
```bash
source activate tf
pip install -r requirements_tf.txt
```

