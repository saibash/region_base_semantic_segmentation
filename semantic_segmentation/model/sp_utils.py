""" functions for TensorFlow layers.
    
    Author:  Loic Landrieu, Martin Simonovsky
    Date:  Dec. 2017 
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869

    Upadted by Jhonatan Contreras
   
"""
from __future__ import division
from __future__ import print_function

import h5py
from sklearn import preprocessing
#from tqdm import tqdm
import numpy as np
import os
import random
#import pcl
import transforms3d
import math
from sklearn.neighbors import KDTree
import copy


import time
import numpy as np
from scipy.spatial.distance import euclidean
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def graipher(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_ind = np.zeros((K,), dtype=int)
    farthest_ind[0] = int(np.random.randint(len(pts)))
    farthest_pts[0] = pts[farthest_ind[0]]

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_ind[i] = int(np.argmax(distances))
        farthest_pts[i] = pts[farthest_ind[i]]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    return farthest_ind, farthest_pts

def augment_cloud(P, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&y, not z
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
    P[:,:3] = np.dot(P[:,:3], M.T)

    if args.pc_augm_jitter:
        sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
    return P


def scaler01(trainlist, testlist, transform_train=True):
    """ Scale edge features to 0 mean 1 stddev """
    edge_feats = np.concatenate([trainlist[i] for i in range(len(trainlist)) ], 0)
    scaler = preprocessing.StandardScaler().fit(edge_feats)

    if transform_train:
        for i in range(len(trainlist)):
            scaler.transform(trainlist[i], copy=False)
    for i in range(len(testlist)):
        scaler.transform(testlist[i], copy=False)
    return trainlist, testlist


def spg_reader(args, fname, incl_dir_in_name=False):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname,'r')
    print(fname)
    if f['sp_labels'].size > 0:
        node_gt_size = f['sp_labels'][:].astype(np.int64)  # column 0: no of unlabeled points, 
                                                           # column 1+: no of labeled points per class
        node_gt = np.argmax(node_gt_size[:, 1:], 1)[:, None]
        node_gt[node_gt_size[:, 1:].sum(1) == 0, :] = -100 # superpoints without labels are to be ignored 
    else:
        N = f['sp_point_count'].shape[0]
        node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N,8), dtype=np.int64)], 1)
        node_gt = np.zeros((N,1), dtype=np.int64)

    node_att = {}
    node_att['xyz'] = f['sp_centroids'][:]
    node_att['nlength'] = np.maximum(0, f['sp_length'][:])
    node_att['volume'] = np.maximum(0, f['sp_volume'][:] ** 2)
    node_att['surface'] = np.maximum(0, f['sp_surface'][:] ** 2)
    node_att['size'] = f['sp_point_count'][:]

    node_att_m = np.concatenate([node_att['xyz'], node_att['nlength'], node_att['volume'], node_att['surface'], node_att['size']],axis=1)

    edges = np.concatenate([ f['source'][:], f['target'][:] ], axis=1).astype(np.int64)
    name = os.path.basename(fname)[:-len('.h5')]
    if incl_dir_in_name: name = os.path.basename(os.path.dirname(fname)) + '/' + name

    return node_att_m, node_gt, node_gt_size, edges, name


def n_sp_reader(fname):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname, 'r')
    n_sp = int(f['sp_centroids'].shape[0])

    return n_sp


def superpoint_reader(args, fname,n_sp, farthest, train):
    """" Load superpoints  """
    hf = h5py.File(fname, 'r')

    superpoint_points = np.zeros((n_sp, args.ptn_npts, 11+3), dtype=np.float32)
    indices = np.ones(n_sp, dtype=np.int32)
   
    for id in range(n_sp):

        P = hf['{:d}'.format(id)]  # xyz, rgb, z, linearity, planarity, scattering, verticality
        N = P.shape[0]
        P = P[:].astype(np.float32)
       
        # np.savetxt("P.txt", P[:,:4], delimiter=' ', fmt='%f %f %f %d' ) 

        rs = np.random.random.__self__ if train else np.random.RandomState(seed=77) # id
      
        if N > args.ptn_npts:  # need to subsample

            ###########################################################################################################
            #                                                                               Statistical Outlier Removal
            ###########################################################################################################
            if args.sor:
                mean_k = args.mean_k
                std_dev = args.std_dev
                array = P[:, :3]

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
                
                # aux = np.ones(np.shape(P[:, :4]))
                # aux[:, :3] = P[:, :3]
                # np.savetxt("no filter segment.txt", aux, delimiter=' ', fmt='%f %f %f %d')  
                P = np.copy(P[idx])
                N = P.shape[0]
                P = P[:].astype(np.float32)

                # aux = np.ones(np.shape(P[:, :4]))
                # aux[:, :3] = P[:, :3]
                # np.savetxt("filter_segment.txt", aux, delimiter=' ', fmt='%f %f %f %d')  

            ############################################################################################################
            if farthest:
                farthest_ind, _ = graipher(P[:, :3], args.ptn_npts)
                P = P[farthest_ind, :]

            else:
                ii = rs.choice(N, args.ptn_npts)
                P = P[ii, ...]

        elif N < args.ptn_npts:  # need to pad by duplication
            ii = rs.choice(N, args.ptn_npts - N)
            P = np.concatenate([P, P[ii, ...]], 0)
            if N < args.ptn_minpts:
                indices[id] = 0

        #  np.savetxt("Original.txt", P[:, :4], delimiter=' ', fmt='%f %f %f %d' )  
        P_copy = copy.deepcopy(P)
        # if not farthest and train:
        if args.pc_augm:
             print("-- \t augment data")
             P[:, :3] = augment_cloud(P[:, :3], args)
            # np.savetxt("Augment.txt", P[:, :4], delimiter=' ', fmt='%f %f %f %d') 
        if args.pc_xyznormalize:
            # normalize xyz into unit ball, i.e. in [-0.5,0.5]
            diameter = np.max(np.max(P[:, :3], axis=0) - np.min(P[:, :3], axis=0))
            P = np.hstack(((P[:, :3] - np.mean(P[:, :3], axis=0, keepdims=True))/ (diameter + 1e-10), P_copy))       
            # np.savetxt("Normalize.txt", P[:, :4], delimiter=' ', fmt='%f %f %f %d') 

        else:
           P = np.hstack(P[:, :3] , P_copy)
     
        superpoint_points[id] = P

    superpoint_points = superpoint_points[indices==1]

    return superpoint_points, indices


def get_datasets(args, farthest=False):

    # ------------------------------------------------------------------------------------------------------------------
    #                                       Reduced 3d
    # ------------------------------------------------------------------------------------------------------------------
    if args.dataset=="reduced3d" or args.dataset=="semantic3d":

        if args.db_train_name == 'training':
            train_names = ['bildstein_station3', 'bildstein_station5', 'domfountain_station2', 'domfountain_station3',
                           'sg27_station4', 'sg27_station1', 'sg27_station9', 'sg27_station2', 'sg28_station4',
                           'untermaederbrunnen_station3', 'domfountain_station1', 'sg27_station5',
                           'untermaederbrunnen_station1', "bildstein_station1"]#, "neugasse_station1"]

            trainset = ['train_reduced/Generic/' + f for f in train_names]

        elif args.db_train_name == 'validation':
            train_names = ['bildstein_station1']
            trainset = ['train_reduced/Generic/' + f for f in train_names]

        elif args.db_test_name == 'validation':

            valid_names = ['bildstein_station1']
            testset = ['train_reduced/Generic/' + f for f in valid_names]

        elif args.db_test_name == 'testing':
            valid_names = ['sg28_Station2_rgb_V2', 'StGallenCathedral_station6_rgb_V2',
                           'MarketplaceFeldkirch_Station4_rgb_V2', 'sg27_station10_rgb_V2']
            testset = ['test_reduced/Generic/' + f for f in valid_names] 

    # ------------------------------------------------------------------------------------------------------------------
    #                                       Kitti-oakland data-set
    # ------------------------------------------------------------------------------------------------------------------
    elif args.dataset == "kitti":

        train_names = []
        trainset = []
        #db_train_name = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6"]
        db_train_name = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/"]

        for area in db_train_name:
            for f in sorted(os.listdir(args.SEMA3D_PATH + '/parsed/' + area + "/Generic/")):
                train_names.append(os.path.splitext(f)[0])
                trainset.append(area + '/Generic/' + os.path.splitext(f)[0])
                #break

        valid_names = []
        testset = []
        db_test_name = ["Area_6/"]

        for area in db_test_name:
            for f in sorted(os.listdir(args.SEMA3D_PATH + '/parsed/' + area + "/Generic/")):
                valid_names.append(os.path.splitext(f)[0])
                testset.append(area + '/Generic/' + os.path.splitext(f)[0])
                #break
    # ------------------------------------------------------------------------------------------------------------------
    #                                            Generic data-set
    # ------------------------------------------------------------------------------------------------------------------

    else:

        train_names = []
        for f in sorted(os.listdir(args.SEMA3D_PATH + '/parsed/' + args.db_train_name + "/Generic/")):
            train_names.append(os.path.splitext(f)[0])
            #break
        trainset = [args.db_train_name + '/Generic/' + f for f in train_names]

        valid_names = []
        for f in sorted(os.listdir(args.SEMA3D_PATH + '/parsed/' + args.db_test_name + "/Generic/")):
                valid_names.append(os.path.splitext(f)[0])
                #break
        testset = [args.db_test_name + '/Generic/' + f for f in valid_names]

    # ------------------------------------------------------------------------------------------------------------------
    #                                           Load superpoints graphs
    # ------------------------------------------------------------------------------------------------------------------
    
    testlist, trainlist = [], []      # store superpoints attributes
    testlist_p, trainlist_p = [], []  # stores points inside superpoints
    test_gt_sp, train_gt_sp = [], []  # store superpoints labels
    train_indices_list, test_indices_list = [], []  # store indices of superpoints with

    for n in trainset:
            tic()
            fname=args.SEMA3D_PATH + 'superpoint_graphs/' + n + '.h5'

            node_att_m, node_gt, node_gt_size, edges, name = spg_reader(args, fname, True)
            superpoint_points, indices = superpoint_reader(args, args.SEMA3D_PATH + "parsed/" + n + '.h5', n_sp_reader(fname), farthest, train=True)

            node_att_m = node_att_m[indices==1]
            node_gt = node_gt[indices==1]

            trainlist.append(node_att_m)
            train_gt_sp.append(node_gt)
            trainlist_p.append(superpoint_points)
            train_indices_list.append(indices)
            toc()

    for n in testset:
            tic()
            fname= args.SEMA3D_PATH + 'superpoint_graphs/' + n + '.h5'
            node_att_m, node_gt, node_gt_size, edges, name = spg_reader(args, fname, True)
            superpoint_points, indices = superpoint_reader(args, args.SEMA3D_PATH + '/parsed/' + n + '.h5', n_sp_reader(fname), farthest, train=False)

            node_att_m = node_att_m[indices==1] 
            node_gt = node_gt[indices==1]

            testlist.append(node_att_m)
            test_gt_sp.append(node_gt)
            testlist_p.append(superpoint_points)
            test_indices_list.append(indices)
            toc()

    # Normalize edge features
    if args.spg_attribs01:

            trainlist_temp = copy.deepcopy(trainlist)
            testlist_temp = copy.deepcopy(testlist)
            trainlist, testlist = scaler01(trainlist, testlist)
            for i in range(len(trainlist_temp)):
                trainlist[i] = np.concatenate(( trainlist[i], np.expand_dims(trainlist_temp[i][:, -1], axis=1)), axis=1)

            for i in range(len(testlist_temp)):
                testlist[i] = np.concatenate(( testlist[i], np.expand_dims(testlist_temp[i][:, -1], axis=1)), axis=1)

    return trainlist_p,train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist, train_names, valid_names, train_indices_list, test_indices_list


