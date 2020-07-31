#!/bin/bash
source activate py36_build
cd segmentation_and_rebuild/partition/ply_c
cmake . -DPYTHON_LIBRARY=$1/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$1/include/python3.6m -DBOOST_INCLUDEDIR=$1/include -DEIGEN3_INCLUDE_DIR=$1/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$1/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$1/include/python3.6m -DBOOST_INCLUDEDIR=$1/include -DEIGEN3_INCLUDE_DIR=$1/include/eigen3
make


