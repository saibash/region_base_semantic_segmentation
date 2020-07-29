"""
    functions for writing and reading features and superpoint graph    
    Author:  Loic Landrieu, Martin Simonovsky
    Date:  Dec. 2017 
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869

    Upadted by Jhonatan Contreras
   
"""
import os
import sys
import random
import glob
from plyfile import PlyData, PlyElement
import numpy as np
import numpy.matlib
from numpy import genfromtxt
import h5py
from sklearn.neighbors import NearestNeighbors

sys.path.append("./cut-pursuit/src")
sys.path.append("./ply_c")
sys.path.append("./partition/cut-pursuit/src")
sys.path.append("./partition/ply_c")
sys.path.append("./learning")
import libply_c
import multiprocessing as mp
import pdal
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#------------------------------------------------------------------------------
def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def geof2ply(filename, xyz, geof):
    """write a ply with colors corresponding to geometric features"""
    color = np.array(255 * geof[:, [0, 1, 3]], dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def prediction2ply(filename, xyz, prediction, n_label):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
        color[np.where(prediction == i_label), :] = get_color_from_label(i_label, n_label)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = {
        'ceiling': 1,
        'floor': 2,
        'wall': 3,
        'column': 4,
        'beam': 5,
        'window': 6,
        'door': 7,
        'table': 8,
        'chair': 9,
        'bookcase': 10,
        'sofa': 11,
        'board': 12,
        'clutter': 13,
        'stairs': 0,
        }.get(object_class, 0)
    return object_label
#------------------------------------------------------------------------------
def get_color_from_label(object_label, n_label):
    """associate the color corresponding to the class"""
    if n_label == 13: #S3DIS
        object_label = {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 233, 229, 107], #'ceiling' .-> .yellow
            2: [  95, 156, 196], #'floor' .-> . blue
            3: [ 179, 116,  81], #'wall'  ->  brown
            4: [  81, 163, 148], #'column'  ->  bluegreen
            5: [ 241, 149, 131], #'beam'  ->  salmon
            6: [  77, 174,  84], #'window'  ->  bright green
            7: [ 108, 135,  75], #'door'   ->  dark green
            8: [  79,  79,  76], #'table'  ->  dark grey
            9: [  41,  49, 101], #'chair'  ->  darkblue
            10: [223,  52,  52], #'bookcase'  ->  red
            11: [ 89,  47,  95], #'sofa'  ->  purple
            12: [ 81, 109, 114], #'board'   ->  grey
            13: [233, 233, 229], #'clutter'  ->  light grey
            }.get(object_label, -1)
    if (object_label==-1):
        raise ValueError('Type not recognized: %s' % (object_label))
    if (n_label == 8): #Semantic3D
        object_label =  {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 200, 200, 200], #'man-made terrain'  ->  grey
            2: [   0,  70,   0], #'natural terrain'  ->  dark green
            3: [   0, 255,   0], #'high vegetation'  ->  bright green
            4: [ 255, 255,   0], #'low vegetation'  ->  yellow
            5: [ 255,   0,   0], #'building'  ->  red
            6: [ 148,   0, 211], #'hard scape'  ->  violet
            7: [   0, 255, 255], #'artifact'   ->  cyan
            8: [ 255,   8, 127], #'cars'  ->  pink
            }.get(object_label, -1)
    if object_label == -1:
        raise ValueError('Type not recognized: %s' % (object_label))
    return object_label
#------------------------------------------------------------------------------
def get_objects(raw_path):
#S3DIS specific
    """extract data from a room folder"""
    room_ver = genfromtxt(raw_path, delimiter=' ')
    xyz = np.array(room_ver[:, 0:3], dtype='float32')
    rgb = np.array(room_ver[:, 3:6], dtype='uint8')
    n_ver = len(room_ver)
    del room_ver
    nn = NearestNeighbors(1, algorithm='kd_tree').fit(xyz)
    room_labels = np.zeros((n_ver,), dtype='uint8')
    room_object_indices = np.zeros((n_ver,), dtype='uint32')
    objects = glob.glob(os.path.dirname(raw_path) + "/Annotations/*.txt")
    i_object = 0
    for single_object in objects:
        object_name = os.path.splitext(os.path.basename(single_object))[0]
        print("        adding object " + str(i_object) + " : "  + object_name)
        object_class = object_name.split('_')[0]
        object_label = object_name_to_label(object_class)
        obj_ver = genfromtxt(single_object, delimiter=' ')
        distances, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
        room_labels[obj_ind] = object_label
        room_object_indices[obj_ind] = i_object
        i_object = i_object + 1
    return xyz, rgb, room_labels, room_object_indices
#------------------------------------------------------------------------------
def write_ply_obj(filename, xyz, rgb, labels, object_indices):
    """write into a ply file. include the label and the object number"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
            , ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')
            , ('object_index', 'uint32')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    vertex_all[prop[7][0]] = object_indices
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def write_ply_labels(filename, xyz, rgb, labels):
    """write into a ply file. include the label"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1')
            , ('blue', 'u1'), ('label', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def write_ply(filename, xyz, rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def read_ply(filename):
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    try:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.float32)
    if np.max(rgb) > 1:
        rgb = rgb
    try:
        object_indices = plydata['vertex']['object_index']
        labels = plydata['vertex']['label']
        return xyz, rgb, labels, object_indices
    except ValueError:
        try:
            labels = plydata['vertex']['label']
            return xyz, rgb, labels
        except ValueError:
            return xyz, rgb
#------------------------------------------------------------------------------
def write_features(file_name, geof, xyz, rgb, graph_nn, labels,dem):
    """write the geometric features, labels and clouds in a h5 file"""
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('linearity', data=geof[:, 0], dtype='float32')
    data_file.create_dataset('planarity', data=geof[:, 1], dtype='float32')
    data_file.create_dataset('scattering', data=geof[:, 2], dtype='float32')
    data_file.create_dataset('verticality', data=geof[:, 3], dtype='float32')
    data_file.create_dataset('source', data=graph_nn["source"], dtype='uint32')
    data_file.create_dataset('target', data=graph_nn["target"], dtype='uint32')
    data_file.create_dataset('distances', data=graph_nn["distances"], dtype='float32')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    data_file.create_dataset('rgb', data=rgb, dtype='uint8')
    data_file.create_dataset('dem', data=dem, dtype='float32')
    if len(labels) > 0 and len(labels.shape)>1 and labels.shape[1]>1:
        data_file.create_dataset('labels', data=labels, dtype='uint32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()
#------------------------------------------------------------------------------
def read_features(file_name):
    """read the geometric features, clouds and labels from a h5 file"""
    data_file = h5py.File(file_name, 'r')
    #fist get the number of vertices
    n_ver = len(data_file["linearity"])
    n_edg = len(data_file["source"])
    has_labels = len(data_file["labels"])
    #the labels can be empty in the case of a test set
    if has_labels:
        #labels = rgb = np.zeros((n_ver, ), dtype='uint8')
        labels = np.array(data_file["labels"])
    else:
        labels = []
    #---create the arrays---
    geof = np.zeros((n_ver, 4), dtype='float32')
    xyz = np.zeros((n_ver, 3), dtype='float32')
    rgb = np.zeros((n_ver, 3), dtype='uint8')
    source = np.zeros((n_edg,), dtype='uint32')
    target = np.zeros((n_edg,), dtype='uint32')
    distances = np.zeros((n_edg,), dtype='float32')
    dem = np.zeros((n_ver, 7), dtype='float32')
    #---fill the arrays---
    geof[:, 0] = data_file["linearity"]
    geof[:, 1] = data_file["planarity"]
    geof[:, 2] = data_file["scattering"]
    geof[:, 3] = data_file["verticality"]
    xyz[:] = data_file["xyz"]
    rgb[:] = data_file["rgb"]
    source[:] = data_file["source"]
    target[:] = data_file["target"]
    distances[:] = data_file["distances"]

    #---set the graph---
    graph_nn = dict([("is_nn", True)])
    graph_nn["source"] = source
    graph_nn["target"] = target
    graph_nn["distances"] = distances
    try:
        dem[:] = data_file["dem"]
        return geof, xyz, rgb, graph_nn, labels, dem
    except:
        return geof, xyz, rgb, graph_nn, labels

#------------------------------------------------------------------------------
def write_spg(file_name, graph_sp, components, in_component):
    """save the partition and spg information"""
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    n_com = len(components)  #

    for i_com in range(0, n_com):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
    data_file.create_dataset('in_component'
                             , data=in_component, dtype='uint32')
    data_file.create_dataset('sp_labels'
                             , data=graph_sp["sp_labels"], dtype='uint32')
    data_file.create_dataset('sp_centroids'
                             , data=graph_sp["sp_centroids"], dtype='float32')
    data_file.create_dataset('sp_length'
                             , data=graph_sp["sp_length"], dtype='float32')
    data_file.create_dataset('sp_surface'
                             , data=graph_sp["sp_surface"], dtype='float32')
    data_file.create_dataset('sp_volume'
                             , data=graph_sp["sp_volume"], dtype='float32')
    data_file.create_dataset('sp_point_count'
                             , data=graph_sp["sp_point_count"], dtype='uint64')
    data_file.create_dataset('source'
                             , data=graph_sp["source"], dtype='uint32')
    data_file.create_dataset('target'
                             , data=graph_sp["target"], dtype='uint32')
    data_file.create_dataset('se_delta_mean'
                             , data=graph_sp["se_delta_mean"], dtype='float32')
    data_file.create_dataset('se_delta_std'
                             , data=graph_sp["se_delta_std"], dtype='float32')
    data_file.create_dataset('se_delta_norm'
                             , data=graph_sp["se_delta_norm"], dtype='float32')
    data_file.create_dataset('se_delta_centroid'
                             , data=graph_sp["se_delta_centroid"], dtype='float32')
    data_file.create_dataset('se_length_ratio'
                             , data=graph_sp["se_length_ratio"], dtype='float32')
    data_file.create_dataset('se_surface_ratio'
                             , data=graph_sp["se_surface_ratio"], dtype='float32')
    data_file.create_dataset('se_volume_ratio'
                             , data=graph_sp["se_volume_ratio"], dtype='float32')
    data_file.create_dataset('se_point_count_ratio'
                             , data=graph_sp["se_point_count_ratio"], dtype='float32')
#-----------------------------------------------------------------------------
def read_spg(file_name):
    """read the partition and spg information"""
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype='uint32')
    graph["target"] = np.array(data_file["target"], dtype='uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
    graph["sp_length"] = np.array(data_file["sp_length"], dtype='float32')
    graph["sp_surface"] = np.array(data_file["sp_surface"], dtype='float32')
    graph["sp_volume"] = np.array(data_file["sp_volume"], dtype='float32')
    graph["sp_point_count"] = np.array(data_file["sp_point_count"], dtype='uint64')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
    graph["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype='float32')
    graph["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype='float32')
    graph["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype='float32')
    graph["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype='float32')
    graph["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype='float32')
    graph["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype='float32')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = len(graph["sp_length"])
    graph["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return graph, components, in_component

#------------------------------------------------------------------------------
def reduced_labels2full(labels_red, unary_labels_red, components, n_ver, n_classes):
    """distribute the labels of superpoints to their respective points"""
    labels_full = np.zeros((n_ver, ), dtype='uint8')
    false_color = np.zeros((n_ver,), dtype='float32')
    segment_id = np.zeros((n_ver,), dtype='float32')

    unary_full = np.zeros((n_ver, np.shape(unary_labels_red)[1]), dtype='float32') # z min, z max, z mean per segment

    for i_com in range(0, len(components)): ## components
        labels_full[components[i_com]] = labels_red[i_com]
        false_color[components[i_com]] = np.random.rand()*100
        segment_id[components[i_com]] = i_com
        unary_full[components[i_com]] = unary_labels_red[i_com]

    return labels_full, false_color, unary_full, segment_id

def build_false_color(components, n_ver, xyz, ver_batch, data_file):
    """distribute the labels of superpoints to their respective points"""
    false_color = np.zeros((n_ver,), dtype='float32')
    for i_com in range(0, len(components)): ## components
        false_color[components[i_com]] = np.random.rand()*1000

    i_rows = 0
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    xyz_txt = np.zeros((0, 3), dtype=np.float32)
    false_color_txt = np.zeros((0, ), dtype='float32')

    while(True):
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
            #vertices[:, 0] += -610000.00
            #vertices[:, 1] += -5700000.00
            #vertices = np.load(data_file)
        except StopIteration:
            # break  # end of file
            print("finish")
            # if np.shape(vertices)[0]<1:
            break
        if np.shape(vertices)[0]<1:
            break

        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        del vertices
        distances, neighbor = nn.kneighbors(xyz_full)

        del distances

        i_rows = i_rows + ver_batch

        false_color_txt = np.hstack((false_color_txt, false_color[neighbor].flatten()))  # false color composition

        xyz_txt = np.concatenate((xyz_txt, xyz_full), axis=0)  # Contain XYZ coordinates + GT accumulative

    false_color_txt = np.concatenate((xyz_txt, np.expand_dims(false_color_txt, axis=1)), axis=1) # XYZ + GT + Falsecolor

    return false_color_txt


#  ------------------------------------------------------------------------------
def prune(data_file, ver_batch, voxel_width, n_class, data_folder, rgb_intensity_index, version="", gt_index=3, label_dict=None):
    """prune the cloud with a regular voxel grid
       label_dict example:  
            labels_rue = {"0": 7, "1": 1, "2": 2, "4": 3, "7": 5, "9": 6, "10": 4, "14":5, "15": 7, "19": 7, "20": 7,
                  "21": 5, "22": 6, "23": 7, "24": 6, "25": 7, "26": 7}

            labels_oak =  {"1004": 1, "1100": 2, "1103": 3, "1200": 4, "1400": 5}

    """
    i_rows = 0
    xyz = np.zeros((0, 3), dtype='float32')
    rgb = np.zeros((0, 3), dtype='uint8')
    labels = np.zeros((0, n_class + 1), dtype='uint32')
    path_dem = data_folder + "Generic/"

    #  ---the clouds can potentially be too big to parse directly---
    #  ---they are cut in batches in the order they are stored---
    while True:
    #if True:
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
            #vertices[:, 0] += -610000.00
            #vertices[:, 1] += -5700000.00


            #vertices = np.load(data_file)
        except StopIteration:
            #break  # end of file
            print("finish")
            break

        if np.shape(vertices)[0]<1:
            break
            
        print(np.shape(vertices)[1])
        if  np.shape(vertices)[1] == 3:  # if has labels
            labels_full = np.zeros(1, dtype='uint8')
            labels = np.zeros(1, dtype='uint8')
            n_class = 0
            if i_rows==0:
               print("has not label on Data")
        else:
            if np.shape(vertices)[1] > gt_index:
               labels_full = np.copy(vertices[:, gt_index]) 
               labels_full = labels_full.astype("uint8")
               print("-- \t has label on Data ")
            else:
               labels_full = np.copy(vertices[:, 3]) 
               labels_full = labels_full.astype("uint8")
               if i_rows==0:
                  print("-- \t warning -- gt_index = {0} out of bounds for {1} columns, used instead gt_index = 3".format(gt_index, np.shape(vertices)[1]))
                

        if not label_dict==None:
                labels_full_t = []
                for item in labels_full:
                    labels_full_t.append(label_dict[str(item)])
                labels_full = np.array(labels_full_t)
                labels_full = labels_full.astype("uint8")
                if i_rows==0:
                   print("-- \t \t has label on Data used directory")

        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        
        if rgb_intensity_index is None:   
           if  np.shape(vertices)[1] >= 6:
               if gt_index==3:
                  rgb_full = np.array(vertices[:, 4:7], dtype='float32')
               elif gt_index==6:
                  rgb_full = np.array(vertices[:, 3:6], dtype='float32')
           else:
               rgb_full = np.ones_like(vertices[:, :3], dtype='float32')

        else:
            
            if  np.shape(vertices)[1] >= np.max(rgb_intensity_index):
              rows, _ = np.shape(vertices)
              rgb_full = np.zeros((rows, 3))
              rgb_full[:, 0] = np.copy(vertices[:, rgb_intensity_index[0]])
              rgb_full[:, 1] = np.copy(vertices[:, rgb_intensity_index[1]])
              rgb_full[:, 2] = np.copy(vertices[:, rgb_intensity_index[2]])
              if i_rows==0:
                 print("has intensity or color ")
            else:
              rgb_full = np.ones_like(vertices[:, :3], dtype='float32')
              if i_rows==0:
                 print("has not intensity or color ")
      

        del vertices

        print("--labels in file section ", np.unique(labels_full))

        xyz_sub, rgb_sub, labels_sub = libply_c.prune(xyz_full, voxel_width, rgb_full, labels_full, n_class)

        del xyz_full, rgb_full
        xyz = np.vstack((xyz, xyz_sub))
        rgb = np.vstack((rgb, rgb_sub))
        labels = np.vstack((labels, labels_sub))
        i_rows = i_rows + ver_batch

        if np.shape(labels_full)[0] < 2:
            labels = []

        ###########################################################################################################
        #                                                                               Statistical Outlier Removal
        ###########################################################################################################
        if False:
            mean_k = 30  # 50# 50 10
            std_dev = 1.0
            array = xyz[:, :3]
            from sklearn.neighbors import KDTree

            # KDTree object (sklearn)
            kDTree = KDTree(array, leaf_size=50)
            dx, idx_knn = kDTree.query(array[:, :], k=mean_k + 1)
            dx, idx_knn = dx[:, 1:], idx_knn[:, 1:]

            distances = np.sum(dx, axis=1) / (mean_k - 1.0)
            valid_distances = np.shape(distances)[0]

            # Estimate the mean and the standard deviation of the distance vector
            sum = np.sum(distances)
            sq_sum = np.sum(distances ** 2)

            mean = sum / float(valid_distances)
            variance = (sq_sum - sum * sum / float(valid_distances)) / (float(valid_distances) - 1)
            stddev = np.sqrt(variance)

            # a distance that is bigger than this signals an outlier
            distance_threshold = mean + std_dev * stddev
            idx = np.nonzero(distances < distance_threshold)
           
            xyz = np.copy(xyz[idx])
            rgb = np.copy(rgb[idx])
            labels = np.copy(labels[idx])

    ####################################################################################################################

    ####################################################################################################################
    xyz_rgb = np.hstack((xyz, rgb))
    file_name = os.path.splitext(os.path.basename(data_file))[0]
    path_short = path_dem + file_name + version

    np.savetxt(path_short + ".txt", xyz_rgb, delimiter=",", header="X,Y,Z,R,G,B", comments='', fmt='%f,%f,%f,%d,%d,%d')
    dem_array = build_dem(path_short, xyz_rgb)

    return xyz, rgb, labels, dem_array


#  ------------------------------------------------------------------------------
def prune_labels(data_file, file_label_path, ver_batch, voxel_width, n_class, data_folder, version=""):
    """prune the cloud with a regular voxel grid - with labels"""
    i_rows = 0
    xyz = np.zeros((0, 3), dtype='float32')
    rgb = np.zeros((0, 3), dtype='uint8')
    labels = np.zeros((0, n_class+1), dtype='uint32')
    path_dem = data_folder + "Generic/"

    # ---the clouds can potentially be too big to parse directly---
    # ---they are cut in batches in the order they are stored---
    while True:
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
            # print(np.shape(vertices))
            # input("SDF")
        except StopIteration:
            # end of file
            break
        # if rgb == False
        print("*")
        if np.shape(vertices)[1] <= 6:  # xyz-rgb
            vertices_tem = np.copy(vertices)
            rows, _ = np.shape(vertices)
            vertices = np.ones((rows, 6))
            vertices[:, :3] = np.copy(vertices_tem[:,:3])

        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        rgb_full = np.array(vertices[:, 4:7], dtype='uint8')

        del vertices
        labels_full = np.genfromtxt(file_label_path, dtype="u1", delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
        xyz_sub, rgb_sub, labels_sub = libply_c.prune(xyz_full, voxel_width, rgb_full, labels_full, n_class)

        del xyz_full, rgb_full
        xyz = np.vstack((xyz, xyz_sub))
        rgb = np.vstack((rgb, rgb_sub))
        labels = np.vstack((labels, labels_sub))
        i_rows = i_rows + ver_batch

    ####################################################################################################################
    
    file_name = os.path.splitext(os.path.basename(data_file))[0]
    path_short = path_dem + file_name + version
    xyz_rgb = np.hstack((xyz, rgb))
    np.savetxt(path_short + ".txt", xyz_rgb, delimiter=",", header="X,Y,Z,R,G,B", comments='')

    #dem_array = build_dem(path_short, xyz_rgb)

    nr, nc = np.shape(xyz_rgb)
    dem_array = np.ones((nr, nc + 1), dtype=np.float32)
    dem_array[:, :3] = xyz_rgb[:, :3]
    dem_array[:, 3] = xyz_rgb[:, 2]
    dem_array[:, 4:] = xyz_rgb[:, 3:]

    ####################################################################################################################

    return xyz, rgb, labels, dem_array


#-----------------------------------------------------------------------------------------------------------------------
def build_dem(path_short, xyz_rgb):
    # using gdal create DEM and DTM
    from string import Template


    input_path_txt = path_short +".txt"

    output_path_las = path_short +".las"

    json = Template("""
    {
        "pipeline": [
            "$input_path_txt",
        {
          "type":"writers.las",
             "dataformat_id":"0",
          "filename":"$output_path_las"
        }
      ]
    }
    }

    """)

    json = json.substitute(input_path_txt=input_path_txt, output_path_las=output_path_las)
    #print(input_path_txt, output_path_las)
    print("Transforming to LAS ...")

    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good
    pipeline.execute()
    # ###################################
    input_path_txt = path_short + ".las"
    output_path_las = path_short + "_ground.las"

    json = Template("""
       {
           "pipeline": [
               "$input_path_txt",
                   {
                    "type":"filters.pmf"
                    },
               {
               "type":"filters.range",
               "limits":"Classification[2:2]"
               },
               {
               "type":"writers.las",
               "dataformat_id":"0",
               "filename":"$output_path_las"
               }
         ]
       }
       """)
    json = json.substitute(input_path_txt=input_path_txt, output_path_las=output_path_las)
    #print(input_path_txt, output_path_las)
    print("Transforming to LAS ...")

    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good

    count = pipeline.execute()
    log = pipeline.log


    # ##################################################################################################################
    dtm_path = path_short + "_dtm.tif"
    # change resolution
    json = Template("""
    {
        "pipeline": [
            "$output_path_las",
            {
                "filename":"$dtm_path",
                "gdaldriver":"GTiff",
                "output_type":"min",
                "resolution":".30",

                "window_size":80,
                "type": "writers.gdal"
            }
        ]
    }

    """)
    # windows size 80
    # "radius":20,
    json = json.substitute(dtm_path=dtm_path,output_path_las=output_path_las)

    pipeline = pdal.Pipeline(json)
    print("Computing DTM ...")
    pipeline.validate()  # check if our JSON and options were good
    count = pipeline.execute()
    log = pipeline.log
    dtm_noise_path = path_short + "_dtm_noise.txt"
    json = Template("""
    {
      "pipeline":[
        {
          "type":"readers.gdal",
          "filename":"$dtm_path"
        },

        {
          "type":"writers.text",
          "filename":"$dtm_noise_path"
        }
      ]
    }
    """)
    json = json.substitute(dtm_path=dtm_path, dtm_noise_path=dtm_noise_path)
    pipeline = pdal.Pipeline(json)
    print("Removing noise and saving final DTM ...")
    pipeline.validate()  # check if our JSON and options were good
    count = pipeline.execute()
    log = pipeline.log

    dtm_noise_path = path_short + "_dtm_noise.txt"
    dtm = np.loadtxt(dtm_noise_path, skiprows=1, delimiter=",")
    dtm3 = dtm[dtm[:, 2] > -100.0]
    dtm_final_path = path_short + "_dtm_final.txt"
    row, _ = np.shape(dtm3)
    np.savetxt(dtm_final_path, np.append(dtm3, np.ones((row, 1)), axis=1))

    ####################################################################################################################
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    print("Computing final DEM")
    del dtm
    dtm = dtm3
    pc = xyz_rgb[:, :3]
    pc2 = np.round(pc, decimals=1)  # considering resolution of 10cm
    dtm = np.round(dtm, decimals=1)
    u, indices = np.unique(dtm[:, :2], axis=0, return_index=True)  # Considering unique points

    new_dtm = dtm[indices, :]
    del indices
    data = {}
    i = 0
    for values in new_dtm[:, 2]:   # dictionary to save dtm 10cm
        data[(str(new_dtm[i, :2]))] = values
        i += 1
    # ------------------------------------------------------------------------------------------------------------------
    # increasing dictionary with lower resolution
    dtm_low = dtm.astype(int)  # np.round(dtm, decimals=0) # considering resolution of 1 m
    pc3 = pc.astype(int)  # np.round(pc, decimals=0)

    u, indices = np.unique(dtm_low[:, :2], axis=0, return_index=True)  # Considering unique pairs of points

    new_dtm_low = dtm_low[indices, :]
    z_temp = dtm[indices, :]
    i = 0
    for values in new_dtm_low[:, 2]:  # dictionary to save dtm 10cm
        try:
            data[(str(new_dtm_low[i, :2]))] = z_temp[i, 2]
            # print (str(new_dtm_low[i, :2]))
            i += 1
        except:
            pass  # already in dictionary

    # ------------------------------------------------------------------------------------------------------------------

    r, c = np.shape(xyz_rgb)
    # -----------------------------------------------------------------------------------------------------------------
    nr, nc = np.shape(xyz_rgb)
    dem_array = np.zeros((nr,nc+1),dtype=np.float32)
    dem_array[:, :3] = xyz_rgb[:, :3]
    dem_array[:, 3] = xyz_rgb[:,  2]
    dem_array[:, 4:] = xyz_rgb[:, 3:]
    for j in range(r):
            try:
                dem_array[j, 3] = dem_array[j, 3] - data[str(pc2[j, :2])]
            except:
                try:
                    dem_array[j, 3] = dem_array[j, 3] - data[str(pc3[j, :2])]
                except:
                    try:
                        dem_array[j, 3] = dem_array[j, 3] - data[str(2+pc3[j, :2])]
                    except:
                        try:
                            dem_array[j, 3] = dem_array[j, 3] - data[str(-2+pc3[j, :2])]
                        except:
                            pass
    np.savetxt(path_short+"_xyz_dem.txt", dem_array) # xyz+ z'+ rgb
    return dem_array


# ------------------------------------------------------------------------------
def interpolate_labels(data_file, data_gt, xyz, labels, false_color, unary, segment_id, n_classes, ver_batch, train=True, gt_index=3, label_dict=None):
    """interpolate the labels of the pruned cloud to the full cloud
    label_dict example:  
            labels_rue = {"0": 7, "1": 1, "2": 2, "4": 3, "7": 7, "9": 6, "10": 4, "14": 5, "15": 5, "19": 7, "20": 7,
                      "21": 7, "22": 7, "23": 7, "24": 7, "25": 7, "26": 7}
        labels_oak = {"1004": 1, "1100": 2, "1103": 3, "1200": 4, "1400": 5}
    """

    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    i_rows = 0
    labels_f = np.zeros((0, ), dtype='uint8')
    #---the clouds can potentially be too big to parse directly---
    #---they are cut in batches in the order they are stored---
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    xyz_txt = np.zeros((0, 4), dtype=np.float32)
   
    label_file = data_gt
    false_color_txt = np.zeros((0, ), dtype='float32')
    segment_id_full = np.zeros((0, ), dtype='float32')

    unary_full = np.zeros((0, np.shape(unary)[1]), dtype='float32')  # z min, z max, z mean
    train = False
    while True:        
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)         
            if train:
                real_label = np.genfromtxt(label_file, max_rows=ver_batch, skip_header=i_rows)
            else:
                 real_label = np.copy(vertices[:, gt_index])         
        except:
            break
        print("*")
        if np.shape(vertices)[0]<1:
            break

        if not label_dict==None:
            labels_full_t = []
            for item in real_label:
                labels_full_t.append(label_dict[str(int(item))])
            labels_full = np.array(labels_full_t)
            labels_full = labels_full.astype("uint8")
            real_label = labels_full.astype("uint8")

        if np.shape(vertices)[0]>2:
            xyz_full = np.array(vertices[:, 0:3], dtype='float32') 
            distances, neighbor = nn.kneighbors(xyz_full)
            del vertices, distances

            labels_f = np.hstack((labels_f, labels[neighbor].flatten()))  # Output Net
            i_rows = i_rows + ver_batch

            false_color_txt = np.hstack((false_color_txt, false_color[neighbor].flatten()))  # false color composition
            segment_id_full = np.hstack((segment_id_full, segment_id[neighbor].flatten()))   # saving segment id

            xyz_full = np.concatenate((xyz_full, np.expand_dims(real_label, axis=1)), axis=1)  # current XYZ + GT batch
            xyz_txt = np.concatenate((xyz_txt, xyz_full), axis=0)  # Contain XYZ coordinates + GT accumulative

            unary_temp = np.squeeze(unary[neighbor])
            unary_full = np.vstack((unary_full, unary_temp))  # saving unary
            del unary_temp
        else:
            break
     
    false_color_txt = np.concatenate((xyz_txt, np.expand_dims(false_color_txt, axis=1)), axis=1) # XYZ + GT + Falsecolor
    xyz_txt = np.concatenate((xyz_txt, np.expand_dims(labels_f, axis=1)), axis=1)  # XYZ + GT + Output

    def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    ##################################################################################################################
    #  smoothing
    # nn_xyz = NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(xyz_txt[:, :3])
    # _, neighbor_xyz = nn_xyz.kneighbors(xyz_txt[:, :3])
    # i = 0
    # while i < 2:
    #    indices_non_zeros = np.where(labels_f == 0)[0]
    #    # print ("a")
    #    new_neighbour = np.isin(neighbor_xyz, indices_non_zeros)
    #    # print ("b")
    #    new_neighbour = np.logical_not(new_neighbour) * neighbor_xyz
    #    # print ("c")
    #    near_neighbour = np.squeeze(first_nonzero(new_neighbour, 1))
    #    # print ("d")
    #    idx = 0
    #    value_near_neighbour = np.zeros(np.shape(near_neighbour), dtype=np.int32)
    #    # print ("e")
    #    for value in neighbor_xyz:
    #        value_near_neighbour[idx] = value[near_neighbour[idx]] # the near neighbour not zero value
    #        idx += 1
    #    # print ("f")
    #    labels_f[indices_non_zeros] = labels_f[value_near_neighbour[indices_non_zeros]]
    #    i += 1
    #    print("iteration : %d" % i)
    #
    #xyz_txt[:, 4] = labels_f
    ##################################################################################################################

    xyz_GT_out_id_prob = np.concatenate((xyz_txt, np.expand_dims(segment_id_full, axis=1), unary_full), axis=1)
    # XYZ + GT + Output + id + probabilities + z min +z max + z_mean
    
    return labels_f, xyz_txt, false_color_txt, xyz_GT_out_id_prob


def process_w(data_file, label_file, ver_batch, train, labels, xyz, false_color, segment_id, unary, i_rows):
        labels_f = np.zeros((0,), dtype='uint8')
        false_color_txt = np.zeros((0,), dtype='float32')
        segment_id_full = np.zeros((0,), dtype='float32')
        unary_full = np.zeros((0, np.shape(unary)[1]), dtype='float32')  # z min, z max, z mean
        xyz_txt = np.zeros((0, 4), dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
        i_rows = i_rows * ver_batch
        print(i_rows)
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
            if train:
                real_label = np.genfromtxt(label_file, max_rows=ver_batch, skip_header=i_rows)
            else:
                real_label = np.ones(np.max(np.shape(vertices)))
        except StopIteration:
            #end of file
            return
        # print("*")
        if np.shape(vertices)[1]<1:
            xyz_full_full = dict()
            xyz_full_full["xyz_txt"] = xyz_txt
            xyz_full_full["labels_f"] = labels_f
            xyz_full_full["segment_id_full"] = segment_id_full
            xyz_full_full["unary_full"] = unary_full
            xyz_full_full["false_color_txt"] = false_color_txt
            # ----------------------------------------------------------------------------------------------------------
            return xyz_full_full


        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        del vertices
        distances, neighbor = nn.kneighbors(xyz_full)

        del distances

        labels_f = np.hstack((labels_f, labels[neighbor].flatten()))  # Output Net

        false_color_txt = np.hstack((false_color_txt, false_color[neighbor].flatten()))  # false color composition
        segment_id_full = np.hstack((segment_id_full, segment_id[neighbor].flatten()))  # saving segment id

        xyz_full = np.concatenate((xyz_full, np.expand_dims(real_label, axis=1)), axis=1)  # current XYZ + GT batch
        xyz_txt = np.concatenate((xyz_txt, xyz_full), axis=0)  # Contain XYZ coordinates + GT accumulative

        # ------------------------------------------------------------------------------------------------------------
        unary_temp = np.squeeze(unary[neighbor])
        unary_full = np.vstack((unary_full, unary_temp))  # saving unary
        del unary_temp

        xyz_full_full = dict()
        xyz_full_full["xyz_txt"] = xyz_txt
        xyz_full_full["labels_f"] = labels_f
        xyz_full_full["segment_id_full"] = segment_id_full
        xyz_full_full["unary_full"] = unary_full
        xyz_full_full["false_color_txt"] = false_color_txt

        # ------------------------------------------------------------------------------------------------------------
        return xyz_full_full


def process_r(results):
    xyz_txt = np.zeros((0, 4), dtype=np.float32)
    labels_f = np.zeros((0,), dtype='uint8')
    segment_id_full = np.zeros((0,), dtype='float32')
    false_color_txt = np.zeros((0,), dtype='float32')

    aux = results[0].get()

    unary_full = np.zeros((0, np.shape(aux["unary_full"])[1]), dtype='float32')  # z min, z max, z mean

    for p in results:
        try:
            xyz_full_full = p.get()
            xyz_txt_t = xyz_full_full["xyz_txt"]
            labels_f_t = xyz_full_full["labels_f"]
            segment_id_full_t = xyz_full_full["segment_id_full"]
            unary_full_t = xyz_full_full["unary_full"]
            false_color_txt_t = xyz_full_full["false_color_txt"]

            xyz_txt = np.concatenate((xyz_txt, xyz_txt_t), axis=0)
            labels_f = np.hstack((labels_f, labels_f_t))
            segment_id_full = np.hstack((segment_id_full, segment_id_full_t))
            false_color_txt = np.hstack((false_color_txt, false_color_txt_t))
            unary_full = np.vstack((unary_full, unary_full_t))
        except:
            print("end")

    return xyz_txt, labels_f, segment_id_full, unary_full, false_color_txt

# ------------------------------------------------------------------------------
def interpolate_labels_mp(data_file, data_gt, xyz, labels, false_color, unary, segment_id, n_classes, ver_batch, train=True):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)

    label_file = data_gt

    # ###################################################################################################
    #  Parallel
    # #################################################################################################
    n_process = 25
    pool = mp.Pool(processes=n_process)
    results = [pool.apply_async(process_w, args=(data_file, label_file, ver_batch, train, labels, xyz,  false_color, segment_id, unary, row)) for row in range(0, n_process)]
    xyz_txt, labels_f, segment_id_full, unary_full, false_color_txt = process_r(results)



    # ####################################################################################################################
    #

    false_color_txt = np.concatenate((xyz_txt, np.expand_dims(false_color_txt, axis=1)),axis=1)  # XYZ + GT + Falsecolor
    xyz_txt = np.concatenate((xyz_txt, np.expand_dims(labels_f, axis=1)), axis=1)  # XYZ + GT + Output

    #  ###############              Just smoothing                             #########################################
    def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    #nn_xyz = NearestNeighbors(n_neighbors=40, algorithm='kd_tree').fit(np.concatenate((xyz_txt[:, :2], np.zeros((np.shape(xyz_txt)[0],1))),axis=1))
    nn_xyz = NearestNeighbors(n_neighbors=40, algorithm='kd_tree').fit(xyz_txt)

    _, neighbor_xyz = nn_xyz.kneighbors(xyz_txt[:, :3])
    # i = 0
    # while i < -2:
    #     indices_non_zeros = np.where(labels_f == 0)[0]
    #     # print ("a")
    #     new_neighbour = np.isin(neighbor_xyz, indices_non_zeros)
    #     # print ("b")
    #     new_neighbour = np.logical_not(new_neighbour) * neighbor_xyz
    #     # print ("c")
    #     near_neighbour = np.squeeze(first_nonzero(new_neighbour, 1))
    #     # print ("d")
    #     idx = 0
    #     value_near_neighbour = np.zeros(np.shape(near_neighbour), dtype=np.int32)
    #     # print ("e")
    #     for value in neighbor_xyz:
    #         value_near_neighbour[idx] = value[near_neighbour[idx]] # the near neighbour not zero value
    #         idx += 1
    #     # print ("f")
    #     labels_f[indices_non_zeros] = labels_f[value_near_neighbour[indices_non_zeros]]
    #     i += 1
    #     print("iteration : %d" % i)
    indices_zero = np.where(labels_f==0)
    labels_f[indices_zero] = 7

    xyz_txt[:, 4] = labels_f
    #  #################################################################################################################
    xyz_GT_out_id_prob = np.concatenate((xyz_txt, np.expand_dims(segment_id_full, axis=1), unary_full), axis=1)
    # XYZ + GT + Output + id + probabilities + z min +z max + z_mean
    print(np.shape(xyz_GT_out_id_prob))

    return labels_f, xyz_txt, false_color_txt, xyz_GT_out_id_prob


#  ------------------------------------------------------------------------------
def prune_ground_rest(data_file, ver_batch, voxel_width, n_class, data_folder, version=""):
    """prune the cloud with a regular voxel grid"""
    i_rows = 0
    xyz = np.zeros((0, 3), dtype='float32')
    rgb = np.zeros((0, 3), dtype='uint8')
    labels = np.zeros((0, n_class + 1), dtype='uint32')
    path_dem = data_folder + "Generic/"

    labels_rue = {"0": 7, "1": 1, "2": 2, "4": 3, "7": 5, "9": 6, "10": 4, "14":5, "15": 7, "19": 7, "20": 7,
                  "21": 5, "22": 6, "23": 7, "24": 6, "25": 7, "26": 7}

    labels_oak =  {"1004": 1, "1100": 2, "1103": 3, "1200": 4, "1400": 5}
    rue = False
    #  ---the clouds can potentially be too big to parse directly---
    #  ---they are cut in batches in the order they are stored---
    while True:
    #if True:
        try:
            vertices = np.genfromtxt(data_file, delimiter=' ', max_rows=ver_batch, skip_header=i_rows)
            #vertices = np.load(data_file)
        except StopIteration:
            #break  # end of file
            print("fail")
        # if np.shape(vertices)[0]<1:
            break
        if np.shape(vertices)[0]<1:
            break


        if np.shape(vertices)[1] >= 7:  # if has labels
            labels_full = np.copy(vertices[:, 6])+1
            labels_full = labels_full.astype("uint8")
            print("has label on Data + RGB")
            print(np.unique(labels_full))
            #input("SDF")
        elif np.shape(vertices)[1] == 4:  # if has labels
            labels_full = np.copy(vertices[:, 3])
            labels_full = labels_full.astype("uint8")

            if rue:
                labels_full_t = []
                for item in labels_full:
                    labels_full_t.append(labels_rue[str(item)])
                labels_full = np.array(labels_full_t)
                labels_full = labels_full.astype("uint8")

            print("has label on Data ")
        elif np.shape(vertices)[1] == 5:  # if has labels
            labels_full = np.copy(vertices[:, 3])

            labels_full_t = []
            for item in labels_full:
                labels_full_t.append(labels_oak[str(int(item))])
            labels_full = np.array(labels_full_t)
            labels_full = labels_full.astype("uint8")

            labels_full = labels_full.astype("uint8")

        else:
            labels_full = np.zeros(1, dtype='uint8')
            labels = np.zeros(1, dtype='uint8')
            n_class = 0
            print("has not label on Data")

        # if rgb == False
        if np.shape(vertices)[1] == 4:  # xyz-rgb
            #  print("no RGB")
            vertices_tem = np.copy(vertices)
            rows, _ = np.shape(vertices)
            vertices = np.ones((rows, 6))
            vertices[:, :3] = np.copy(vertices_tem[:, :3])


        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        rgb_full = np.array(vertices[:, 3:6], dtype='uint8')#3:6

        del vertices

        if rue:
            indices = labels_full < 7
            xyz_full = xyz_full[indices, :]
            rgb_full = rgb_full[indices, :]
            labels_full = labels_full[indices]

        kitti = True
        if kitti:
            indices = labels_full < 14
            xyz_full = xyz_full[indices, :]
            rgb_full = rgb_full[indices, :]
            labels_full = labels_full[indices]

        print(np.unique(labels_full))

        xyz_sub, rgb_sub, labels_sub = libply_c.prune(xyz_full, voxel_width, rgb_full, labels_full, n_class)

        del xyz_full, rgb_full
        xyz = np.vstack((xyz, xyz_sub))
        rgb = np.vstack((rgb, rgb_sub))
        labels = np.vstack((labels, labels_sub))
        i_rows = i_rows + ver_batch

        if np.shape(labels_full)[0] < 2:
            labels = []

        ###########################################################################################################
        #                                                                               Statistical Outlier Removal
        ###########################################################################################################
        if False:
            mean_k = 30  # 50# 50 10
            std_dev = 1.0
            array = xyz[:, :3]
            from sklearn.neighbors import KDTree

            # KDTree object (sklearn)
            kDTree = KDTree(array, leaf_size=50)
            dx, idx_knn = kDTree.query(array[:, :], k=mean_k + 1)
            dx, idx_knn = dx[:, 1:], idx_knn[:, 1:]

            distances = np.sum(dx, axis=1) / (mean_k - 1.0)
            valid_distances = np.shape(distances)[0]

            # Estimate the mean and the standard deviation of the distance vector
            sum = np.sum(distances)
            sq_sum = np.sum(distances ** 2)

            mean = sum / float(valid_distances)
            variance = (sq_sum - sum * sum / float(valid_distances)) / (float(valid_distances) - 1)
            stddev = np.sqrt(variance)

            # a distance that is bigger than this signals an outlier
            distance_threshold = mean + std_dev * stddev
            idx = np.nonzero(distances < distance_threshold)
            # print(np.shape(P), N)
            # aux = np.ones(np.shape(P[:, :4]))
            # aux[:, :3] = P[:, :3]
            # np.savetxt("no filter segment.txt", aux, delimiter=' ', fmt='%f %f %f %d')  # X is an array

            xyz = np.copy(xyz[idx])
            rgb = np.copy(rgb[idx])
            labels = np.copy(labels[idx])


            # aux = np.ones(np.shape(P[:, :4]))
            # aux[:, :3] = P[:, :3]
            # np.savetxt("filter_segment.txt", aux, delimiter=' ', fmt='%f %f %f %d')  # X is an array
            # print(np.shape(P), N)
            ############################################################################################################

    ####################################################################################################################
    xyz_rgb = np.hstack((xyz, rgb))
    file_name = os.path.splitext(os.path.basename(data_file))[0]
    path_short = path_dem + file_name + version

    np.savetxt(path_short + ".txt", xyz_rgb, delimiter=",", header="X,Y,Z,R,G,B", comments='', fmt='%f,%f,%f,%d,%d,%d')

    dem_array, xyz_rgb = build_dem_ground_rest(path_short, xyz_rgb)
    print(np.shape(rgb), rgb[0, :])
    ####################################################################################################################

    indices = dem_array[:,3]>.2

    xyz = xyz[indices, :]
    rgb = rgb[indices,:]
    labels = labels[indices]
    dem_array = dem_array[indices,:]


    return xyz, rgb, labels, dem_array


#-----------------------------------------------------------------------------------------------------------------------
def build_dem_ground_rest(path_short, xyz_rgb):
    # using gdal create DEM and DTM
    from string import Template

    input_path_txt = path_short +".txt"
    output_path_las = path_short +".las"

    json = Template("""
    {
        "pipeline": [
            "$input_path_txt",
        {
          "type":"writers.las",
             "dataformat_id":"0",
          "filename":"$output_path_las"
        }
      ]
    }
    }

    """)

    json = json.substitute(input_path_txt=input_path_txt, output_path_las=output_path_las)
    print(input_path_txt, output_path_las)
    print("Transforming to LAS ...")

    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good
    pipeline.execute()
    # ###################################
    input_path_txt = path_short + ".las"
    output_path_las = path_short + "_ground.las"

    json = Template("""
       {
           "pipeline": [
               "$input_path_txt",
                   {
                    "type":"filters.pmf"
                    },
               {
               "type":"filters.range",
               "limits":"Classification[2:2]"
               },
               {
               "type":"writers.las",
               "dataformat_id":"0",
               "filename":"$output_path_las"
               }
         ]
       }
       """)
    json = json.substitute(input_path_txt=input_path_txt, output_path_las=output_path_las)
    print(input_path_txt, output_path_las)
    print("Transforming to LAS ...")

    pipeline = pdal.Pipeline(json)
    pipeline.validate()  # check if our JSON and options were good

    count = pipeline.execute()
    log = pipeline.log

    # ##################################################################################################################
    dtm_path = path_short + "_dtm.tif"
    # change resolution
    json = Template("""
    {
        "pipeline": [
            "$output_path_las",
            {
                "filename":"$dtm_path",
                "gdaldriver":"GTiff",
                "output_type":"min",
                "resolution":".30",

                "window_size":80,
                "type": "writers.gdal"
            }
        ]
    }

    """)
    # windows size 80
    # "radius":20,
    json = json.substitute(dtm_path=dtm_path,output_path_las=output_path_las)

    pipeline = pdal.Pipeline(json)
    print("Computing DTM ...")
    pipeline.validate()  # check if our JSON and options were good
    count = pipeline.execute()
    log = pipeline.log
    dtm_noise_path = path_short + "_dtm_noise.txt"
    json = Template("""
    {
      "pipeline":[
        {
          "type":"readers.gdal",
          "filename":"$dtm_path"
        },

        {
          "type":"writers.text",
          "filename":"$dtm_noise_path"
        }
      ]
    }
    """)
    json = json.substitute(dtm_path=dtm_path, dtm_noise_path=dtm_noise_path)
    pipeline = pdal.Pipeline(json)
    print("Removing noise and saving final DTM ...")
    pipeline.validate()  # check if our JSON and options were good
    count = pipeline.execute()
    log = pipeline.log

    dtm_noise_path = path_short + "_dtm_noise.txt"
    dtm = np.loadtxt(dtm_noise_path, skiprows=1, delimiter=",")
    dtm3 = dtm[dtm[:, 2] > -100.0]
    dtm_final_path = path_short + "_dtm_final.txt"
    row, _ = np.shape(dtm3)
    np.savetxt(dtm_final_path, np.append(dtm3, np.ones((row, 1)), axis=1))

  
    # ------------------------------------------------------------------------------------------------------------------
    print("Computing final DEM")
    del dtm
    dtm = dtm3
 
    pc = xyz_rgb[:, :3]
    pc2 = np.round(pc, decimals=1)  # considering resolution of 10cm
    dtm = np.round(dtm, decimals=1)
    u, indices = np.unique(dtm[:, :2], axis=0, return_index=True)  # Considering unique points

    new_dtm = dtm[indices, :]
    del indices
    data = {}
    i = 0
    for values in new_dtm[:, 2]:   # dictionary to save dtm 10cm
        data[(str(new_dtm[i, :2]))] = values
        i += 1
    # ------------------------------------------------------------------------------------------------------------------
    # increasing dictionary with lower resolution
    dtm_low = dtm.astype(int)  # np.round(dtm, decimals=0) # considering resolution of 1 m
    pc3 = pc.astype(int)  # np.round(pc, decimals=0)

    u, indices = np.unique(dtm_low[:, :2], axis=0, return_index=True)  # Considering unique pairs of points

    new_dtm_low = dtm_low[indices, :]
    z_temp = dtm[indices, :]
    i = 0
    for values in new_dtm_low[:, 2]:  # dictionary to save dtm 10cm
        try:
            data[(str(new_dtm_low[i, :2]))] = z_temp[i, 2]
            # print (str(new_dtm_low[i, :2]))
            i += 1
        except:
            pass  # already in dictionary

    # ------------------------------------------------------------------------------------------------------------------

    r, c = np.shape(xyz_rgb)
    # -----------------------------------------------------------------------------------------------------------------
    nr, nc = np.shape(xyz_rgb)
    dem_array = np.zeros((nr,nc+1),dtype=np.float32)
    dem_array[:, :3] = xyz_rgb[:, :3]
    dem_array[:, 3] = xyz_rgb[:,  2]
    dem_array[:, 4:] = xyz_rgb[:, 3:]
    for j in range(r):
            try:
                dem_array[j, 3] = dem_array[j, 3] - data[str(pc2[j, :2])]
            except:
                try:
                    dem_array[j, 3] = dem_array[j, 3] - data[str(pc3[j, :2])]
                except:
                    try:
                        dem_array[j, 3] = dem_array[j, 3] - data[str(2+pc3[j, :2])]
                    except:
                        try:
                            dem_array[j, 3] = dem_array[j, 3] - data[str(-2+pc3[j, :2])]
                        except:
                            pass
    np.savetxt(path_short+"_xyz_dem.txt", dem_array) # xyz+ z'+ rgb
    return dem_array, xyz_rgb

