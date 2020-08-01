#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Point Cloud Unsupervised Segmentation
    Author:  Loic Landrieu, Martin Simonovsky
    Date:  Dec. 2017 
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    Upadted by Jhonatan Contreras
   
"""
import os.path
import glob
import sys
import numpy as np
import argparse
from graphs import *
from provider import *
from timeit import default_timer as timer
sys.path.append("./partition/cut-pursuit/src")
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c
import h5py
import parser_build
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Point Cloud Unsupervised Segmentation')
# --------------------------------------------                 parameters     ------------------------------------------
parser.add_argument('--path_to_data', default="/Data/forest4D_dlr", help="Path to data") 
parser.add_argument('--path_to_output', default="Data/forest4D_dlr", help="Path to output") 
parser.add_argument('--n_labels', default=4, type=int, help='number of classes')  
parser.add_argument('--areas', default="testing/, training/, validation/", help="list of subfolders to be processed separated by ( , ) ")
parser.add_argument('--RGB', default="False", help='True if Data set contains RGB information')
parser.add_argument('--version', default="V0", help='for multiples segmentation parameters '
                                                    'use a different version output name')

parser.add_argument('--file_extension', default=".txt", help='file extension default txt')
parser.add_argument('--gt_index', default=3, type=int, help='ground true index in file')
parser.add_argument('--rgb_intensity_index', type=list, help='rgb or intensity index in file')


parser.add_argument('--ver_batch', default=2000000, type=int, help='Batch size for reading large files')
parser.add_argument('--voxel_width', default=0.01, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=.1, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')

parser.add_argument('--sorted', default=False, type=bool, help="Reverse order to read the data")
parser.add_argument('--overwrite', default=False, type=bool, help="Consider previous results")
parser.add_argument('--print_progress', default=True, type=bool, help="Print current process")


args = parser.parse_args()
rgb_intensity_index = args.rgb_intensity_index

if not rgb_intensity_index==None:
   for i in range(0, len(rgb_intensity_index)): 
       rgb_intensity_index[i] = int(rgb_intensity_index[i]) 
   
   if not (len(rgb_intensity_index) == 1 or len(rgb_intensity_index)==3):
      raise ValueError(' \t error, index len must be for rgb = 3 or for intensity = 1, given = %d' % len(rgb_intensity_index))
   if len(rgb_intensity_index) == 1:
      rgb_intensity_index = np.tile(rgb_intensity_index, 3)
      print(rgb_intensity_index)


args.sorted = False
if not (args.file_extension == ".txt" or args.file_extension == ".npy"):
  raise ValueError('data file not supported', args.file_extension)


root = args.path_to_data + "/"
output_path = args.path_to_output + "/"
#areas = ["test_reduced/", "test_full/", "train/"]
mylist = args.areas
areas = mylist.split(',')

#------------------------------------------------------------------------------
num_area = len(areas)
times = [0, 0, 0]
if not os.path.isdir(root +  "data"):
    print("No data or directory:", root +  "data" )
    #raise ValueError
    sys.exit()
else:
    for area in areas:
        if not os.path.isdir(root +  "data/" + area):
           print("-- Directory not found:\t", root +  "data/" + area )
           sys.exit()
        else:
           print("-- Directory found:\t", root +  "data/" + area )

if not os.path.isdir(output_path):
    os.mkdir(output_path)
    os.mkdir(output_path + "data")
    os.mkdir(output_path + "features")
    os.mkdir(output_path + "superpoint_graphs")
    os.mkdir(output_path + "parsed")
    os.mkdir(output_path + "labels")


if not os.path.isdir(output_path + "data"):
    os.mkdir(output_path + "data")
 
if not os.path.isdir(output_path + "features"):
    os.mkdir(output_path + "features")

if not os.path.isdir(output_path + "superpoint_graphs"):
    os.mkdir(output_path + "superpoint_graphs")

if not os.path.isdir(output_path + "parsed"):
    os.mkdir(output_path + "parsed")

if not os.path.isdir(output_path + "labels"):
    os.mkdir(output_path + "labels")
  
for area in areas:

    if not os.path.isdir(output_path + "data/" + area):
        os.mkdir(output_path + "data/" + area)
    if not os.path.isdir(output_path + "data/" + area + "/Generic/"):
        os.mkdir(output_path + "data/" + area +"/Generic/" )

    if not os.path.isdir(output_path + "features/"+ area):
        os.mkdir(output_path + "features/"+ area)
    if not os.path.isdir(output_path + "features/"+ area + "/Generic/" ):
        os.mkdir(output_path + "features/" + area + "/Generic/" )

    if not os.path.isdir(output_path + "superpoint_graphs/" + area):
        os.mkdir(output_path + "superpoint_graphs/" + area)
    if not os.path.isdir(output_path + "superpoint_graphs/" + area + "/Generic/" ):
        os.mkdir(output_path + "superpoint_graphs/" + area + "/Generic/" )

    if not os.path.isdir(output_path + "parsed/" + area):
        os.mkdir(output_path + "parsed/" + area)
    if not os.path.isdir(output_path + "parsed/" + area + "/Generic/" ):
        os.mkdir(output_path + "parsed/" + area + "/Generic/" )
    
    if not os.path.isdir(output_path + "labels/" + area):
        os.mkdir(output_path + "labels/" + area)
    if not os.path.isdir(output_path + "labels/" + area + "/Generic/" ):
        os.mkdir(output_path + "labels/" + area + "/Generic/" )


if True:
 for area in areas:
    #continue
    print("=================\n   " + area + "\n=================")
    data_folder = root + "data/" + area
    output_folder = output_path + "data/" + area
    fea_folder = output_path + "features/" + area + "Generic/"
    spg_folder = output_path + "superpoint_graphs/" + area + "Generic/"

    files = sorted(glob.glob(data_folder+"*"+args.file_extension), reverse=args.sorted)
    n_files = len(files)
    i_file = 0

    if n_files == 0:
        raise ValueError('%s is empty' % data_folder)

    for file in files:
    
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_short = '_'.join(file_name.split('_')[:3])  
        data_file = data_folder + file_name + args.file_extension
        label_file = data_folder + file_name + ".labels"
        if args.version == "V0" or args.version == "v0":
            fea_file = fea_folder + file_name_short + '.h5'
            spg_file = spg_folder + file_name_short + '.h5'         
        else:
            fea_file = fea_folder + file_name_short + "_" + args.version + '.h5'
            spg_file = spg_folder + file_name_short + "_" + args.version + '.h5'
          
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> " + file_name_short)
        #  ------------------------- build the geometric feature file h5 file ------------------------------------
        if os.path.isfile(fea_file) and args.overwrite:
            if args.print_progress:
                print("-- \t reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels, dem = read_features(fea_file)
            has_labels = len(labels) > 0
            print("-- \t unique classes in file: \t", np.unique(labels))

        else:
            if args.print_progress:
                print("-- \t creating the feature file...")
            start = timer()
            has_labels = (os.path.isfile(label_file))
            # ---------------------  retrieving and sub-sampling the point clouds ---------------------------------
            if has_labels:
                print("-- \t has label file")
                xyz, rgb, labels, dem = prune_labels(data_file, label_file, args.ver_batch, args.voxel_width,
                                                     args.n_labels, output_folder, args.version)
            else:
                print("-- \t has not label file")
                xyz, rgb, labels, dem = prune(data_file, args.ver_batch, args.voxel_width, args.n_labels, output_folder, rgb_intensity_index, args.version, args.gt_index)

            print("-- \t unique classes in file: \t", np.unique(labels))

            # ----------------------  computing the nn graphs ----------------------------------------------------
            if args.print_progress:            
               print("-- \t computing the nn graphs...")
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
           
            # -----------------------  compute geometric features ------------------------------------------------
            if args.print_progress: 
                print("-- \t compute geometric features...")
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            if args.print_progress:
               print("-- \t writing feature file...")
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels, dem)
            if args.print_progress:
               print("-- \t done...")
            
        # --------------------------  compute the partition  ------------------------------------------------------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and args.overwrite:
            if args.print_progress:
                print("-- \t reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            if args.print_progress:
                print("-- \t computing the superpoint graph...")
            # ---------------------- build the spg h5 file ---------------------------------------------------------
            start = timer()

            if args.RGB=="True":
                features = np.hstack((geof, rgb[:,:3] / 255., rgb[:,3:6] )).astype('float32')  # add rgb as a feature for partitioning
                print("-- RGB in segmentation")
            else:
                print("-- No RGB in segmentation")
                features = geof

            features[:, 3] = 2. * features[:, 3]  # increase importance of verticality (heuristic)
            graph_nn["edge_weight"] = np.array(1. / (args.lambda_edge_weight + graph_nn["distances"] /
                                               np.mean(graph_nn["distances"])), dtype='float32')
            if args.print_progress:
                print("-- \t minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                                        , graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype='object')
            end = timer()
            times[1] = times[1] + end - start
            if args.print_progress:
                print("-- \t computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, args.n_labels, dem)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)
        if args.print_progress:
            print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))

        #  Saving false color composition
        n_ver = xyz.shape[0]
        labels_red = np.zeros((n_ver, 0))
        false_color_txt = build_false_color(components, n_ver, xyz, args.ver_batch, data_file)

        false_color_file = output_folder + "Generic/" + file_name_short + '_false_color' + args.version + ".txt"
        np.savetxt(false_color_file, false_color_txt, delimiter=' ', fmt='%f %f %f %d')  # X is an array

parser_build.preprocess_pointclouds(output_path, areas)

