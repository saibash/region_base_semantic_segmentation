#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    call this function once the partition and inference was made to upsample
    the prediction to the original point clouds  
    Author:  Loic Landrieu, Martin Simonovsky
    Date:  Dec. 2017 
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869

    Upadted by Jhonatan Contreras
   
"""
   
import argparse
import numpy as np
import os.path
import glob
from provider import *
import metric


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_metric(output, n_label):
    print("\n#############################################################################################")
    print("                                         Metrics ")
    print("#############################################################################################\n")
    indices_ = output[:, 4] > 0
    output = output[indices_]
    # just consider classified points
    gt = np.array(output[:, 3] - 1).astype(int) 
    y = np.array(output[:, 4] - 1).astype(int) 

    print("Unique GT     : ", 1 + np.unique(gt))
    print("Unique output : ", 1 + np.unique(y))
    results = metric.ConfusionMatrix(number_of_labels=n_label)
    results.count_predicted_batch(ground_truth_vec=gt, predicted=y)
    n_label_partial = int(len(np.unique(gt)))
    print("IoU per class:", [" %.3f " % e for e in results.get_intersection_union_per_class()])
    #print("IoU mean : %.3f" % results.get_average_intersection_union())
    print("IoU mean : %.3f" % (results.get_average_intersection_union()*n_label/n_label_partial))
    print("OA : %.3f" % results.get_overall_accuracy())
    print("mean Acc : %.3f" % results.get_mean_class_accuracy())
    precision, recall, f1 = results.get_precision()
    print("Precision : ", [" %.3f" % e for e in precision])
    print("Recall : ", [" %.3f" % e for e in recall])
    print("F1 : ", [" %.3f" % e for e in f1])
    print("Recall : %.3f" % (sum(recall) / n_label_partial))
    print("Precision : %.3f" % (sum(precision) / n_label_partial))
    print("mean F1 : %.3f" % (sum(f1) / n_label_partial))


parser = argparse.ArgumentParser(description='Write segments, Point Cloud Semantic Segmentation with Edge Convolutions')
parser.add_argument('--dataset', default="default", help="default or others")
parser.add_argument('--path_to_data', default='/Data/forest4D_dlr') 
parser.add_argument('--path_to_output', default='/Data/forest4D_dlr', help='Directory to store results')
parser.add_argument('--ver_batch', default=500000, type=int, help='Batch size for reading large files')
parser.add_argument('--db_test_name', default='training') 
parser.add_argument('--areas', default="testing/, training/, validation/", help="areas to be processed")
parser.add_argument('--n_classes', default=4, type=int, help='number of classes') 
parser.add_argument('--metrics', default=False, type=str2bool, help='Compute metrics') 
parser.add_argument('--file_extension', default=".txt", help='file extension default txt')
parser.add_argument('--gt_index', default=3, type=int, help='ground true index in file')
args = parser.parse_args()

#  ---path to data---------------------------------------------------------------
# root of the data directory and output directory
root = args.path_to_data + '/'
root2 = args.path_to_output + '/'
mylist = args.areas
areas = mylist.split(',')
num_area = len(areas)
n_classes = args.n_classes

if not os.path.isdir(root +  "data"):
    print("No data or directory:", root +  "data" )
    #raise ValueError
    sys.exit()
else:
    for area in areas:
        if not os.path.isdir(root +  "data/" + area):
           print("-- Directory not found:\t", root +  "data/" + area )
           sys.exit()

for area in areas:
        if not os.path.isdir(root2 +  "labels/" + area):
           if not os.path.isdir(root2 +  "labels/"):
              os.mkdir(root2 +  "labels/")
           os.mkdir(root2 +  "labels/" + area)
for area in areas:
      
    # ------------------------------------------------------------------------------
    print("=================\n   " + area + "\n=================")
    data_folder = root  + "data/" + area
    fea_folder  = root2 + "features/" + area + "Generic/"
    spg_folder  = root2 + "superpoint_graphs/" + area + "Generic/"
    res_folder  = args.path_to_output + '/results/'
    labels_folder = root2 + "labels/" + area + "Generic/"
    if not os.path.isdir(data_folder):
       raise ValueError("%s do not exists" % data_folder)
    if not os.path.isdir(fea_folder):
       raise ValueError("%s do not exists" % fea_folder)
    if not os.path.isdir(res_folder):
       raise ValueError("%s do not exists" % res_folder)  
    if not os.path.isdir(root + "labels/"):
       os.mkdir(root + "labels/")   
    if not os.path.isdir(labels_folder):
       os.mkdir(labels_folder)   

    files = sorted(glob.glob(data_folder + "*" + args.file_extension), reverse=False)

    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)

    n_files = len(files)
    i_file = 0
    for file in files:
        #if input(file+ "==> 1 to compute : ")=="1":
        file_name = os.path.splitext(os.path.basename(file))[0]
        if args.dataset == "default":
           a = 3  
           b = 5
        else:
           a = 3
           b = 6
        file_name_short = '_'.join(file_name.split('_')[:a]) #+"_V1" # 2 for semantic 3d

        data_file = data_folder + file_name + args.file_extension
        file_name_gt = os.path.splitext(os.path.basename(file))[0]
        file_name_gt = "_".join(file_name_gt.split("_")[:b])

        data_gt = root + "data/" + area + file_name_gt + ".labels"
        fea_file = fea_folder + file_name_short + '.h5'
        spg_file = spg_folder + file_name_short + '.h5'
        label_file = labels_folder + file_name_short + ".labels"
        xyz_label_file = labels_folder + file_name_short + ".txt"
        xyz_label_file_short = labels_folder + file_name_short + 'short' + ".txt"
        false_color_file = labels_folder + file_name_short + 'false_color' + ".txt"
        xyz_GT_out_id_prob_file = labels_folder + file_name_short + "xyz_GT_out_id_prob.txt"
        i_file = i_file + 1

        files_ready = glob.glob(labels_folder + "*" + args.file_extension )
        if xyz_label_file in files_ready:
           print("ready...")
           overwrite = "1"  #input("To overwrite files type 1 : ")
        else:
           overwrite = "1"
        if overwrite == "1":
        
            print(str(i_file) + " / " + str(n_files) + "---> "+file_name_short)
            print("-- \t reading the subsampled file...")
            geof, xyz, rgb, graph_nn, l, dem = read_features(fea_file)
            graph_sp, components, in_component = read_spg(spg_file)
            n_ver = xyz.shape[0]
            del geof, rgb, graph_nn, l, graph_sp
           
            unary_res_file = h5py.File(res_folder + 'unary_predictions_' + file_name_short + '.h5', 'r')
            res_file = h5py.File(res_folder + 'predictions_' + file_name_short + '.h5', 'r')
            labels_red = np.array(res_file.get(file_name_short))

            unary_labels_red = np.array(unary_res_file.get(file_name_short))

            labels_full, false_color, unary_full, segment_id = reduced_labels2full(labels_red, unary_labels_red,
                                                                                   components, n_ver, n_classes)
            print("-- \t interpolate labels...")
            labels_ups, xyz_label, false_color_txt, xyz_GT_out_id_prob = interpolate_labels(data_file, 
                                                                         data_gt, xyz, labels_full, false_color,
                                                                         unary_full, segment_id, n_classes, 
                                                                         args.ver_batch, train=False,
                                                                         gt_index=args.gt_index)
            print("-- \t saving files...")
            np.savetxt(label_file, labels_ups, delimiter=' ', fmt='%d')   
            np.savetxt(xyz_label_file, xyz_label, delimiter=' ', fmt='%f %f %f %d %d')  
            np.savetxt(false_color_file, false_color_txt, delimiter=' ', fmt='%f %f %f %d %d')  

            b = "%.3f "
            fmt = "%f %f %f %d %d %d  "
            for i in range(n_classes):
                fmt += b
            fmt += '%.3f %.3f %.3f'

        
            # saving short version of gt
            indices = xyz_label[:, 3] <= 12
            xyz_label = xyz_label[indices]

            np.savetxt(xyz_label_file_short, xyz_label, delimiter=' ', fmt='%f %f %f %d %d')
            xyz_GT_out_id_prob = xyz_GT_out_id_prob[indices]
            print("-- \t shape xyz + GT + out + probabilities : ", np.shape(xyz_GT_out_id_prob))
            np.savetxt(xyz_GT_out_id_prob_file, xyz_GT_out_id_prob, delimiter=' ', fmt=fmt)
            print("-- \t saved")

            if args.metrics:
               print_metric(xyz_label, n_classes)  

