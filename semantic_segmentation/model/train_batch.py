"""
################################################################################
This code aims to develop a 3D semantic segmentation


author    : Jhonatan Contreras
copyright : University of Jena

ABOUT:
    Contains the TrainModel class. Which contains all the necessary code
    to Create a Tensorflow graph, and training operations.
    It is a parent class for TestModel class
################################################################################
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import numpy as np
import os
import glob as glob
import h5py
import math
import metric
import math
from tqdm import tqdm
import argparse
from model import sp_utils
from utils import tf_util
from multiprocessing import Pool
from utils import tf_util
from multiprocessing import Pool
import copy
from itertools import islice
import time

from dgcnn_pack import Seseg_basic as model
from dgcnn_pack import dgcnn_basic as dgcnn


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # "0,1" -1 for not GPU


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#       for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#       print(e)
#
# def unwrap_self_pool_train(arg, **kwarg):
#     return segmentNet.pool_train(*arg, **kwarg)

#
# def unwrap_self_load_again(arg, **kwarg):
#     return segmentNet.load_again(*arg, **kwarg)


def load_again(args):
    load = dict()
    trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist, _, _, train_indices_list, test_indices_list = sp_utils.get_datasets(args)
    load["trainlist_p"] = trainlist_p
    load["train_gt_sp"] = train_gt_sp
    load["trainlist"] = trainlist
    load["testlist_p"] = testlist_p
    load["test_gt_sp"] = test_gt_sp
    load["testlist"] = testlist
    load["train_indices_list"] = train_indices_list
    load["test_indices_list"] = test_indices_list
    print("  -- load Done --")
    return load



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


class SuperPointNet(object):

    def __init__(self, trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist, train_indices_list,
                 test_indices_list, args, class_weights):
        """Initializes Class"""
        self.args = args
        self.txt_name = args.odir+'/'+ args.log_file

        if args.rgb:
            self.rgb = 6
        else:
            self.rgb = 3
        self.learning_rate = args.learning_rate
        self.class_weights = class_weights
        self.graph = tf.Graph()
        # MODEL SETTINGS
        self.batch_size = args.batch_size
        self.data_size = args.batch_size 
        self.n_ps = args.ptn_npts  # number of points in each segment
        self.n_samples = np.shape(trainlist_p)[0]  # number of training samples
        self.n_samples_testing = np.shape(testlist_p)[0]  # number of training samples
        self.feature_len = 15 # number of attributes of segments       
        self.trainlist_p = trainlist_p  # numpy array store points inside each segment
        self.train_gt = train_gt_sp     # label information for all segments (-100 for unlabels)
        self.trainlist = trainlist      # list of attributes for the hole segment
        self.train_indices_list = train_indices_list  # store indices of segments
        self.test_indices_list = test_indices_list    # store indices of segments
        self.testlist_p = testlist_p  # numpy array store points inside each segment for testing
        self.test_gt = test_gt_sp     # label information for all segments (-100 for unlabels) for testing
        self.testlist = testlist      # list of attributes for the hole segment for testing

        self.resume = args.resume
        self.restoring_parameters = args.restoring_partially
        self.resume_best_val = args.resume_best_val
        if self.restoring_parameters:
           self.resume = True
        if self.resume_best_val:
           self.resume = True
        # DIRECTORIES TO STORE PRELIMINARY RESULTS
        self.snapshot_dir = args.odir + "/snapshots/" + args.model_name +"/backup_class_weight"
        self.snapshot_file = os.path.join(self.snapshot_dir, "snapshot.chk")
        self.tensorboard_dir = args.odir +  "/tensorboard/" + args.model_name
        self.num_classes = args.n_classes  # eight classes
        # DIRECTORIES TO LOAD PRE TRAIN PARAMETERS
        self.snapshot_dir_pre =args.odir + "/snapshots/" + args.pre_train + "/backup_class_weight"
        self.snapshot_file_pre = os.path.join(self.snapshot_dir_pre, "snapshot.chk")
        if not os.path.exists(self.args.odir):
            os.makedirs(self.args.odir)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        if args.restoring_partially:
           if not os.path.exists(self.snapshot_dir_pre):
              self.snapshot_dir_pre = self.snapshot_dir
              self.snapshot_file_pre = self.snapshot_file
              print("Restoring_partially fail, Not pre train folder found")
        else:
            self.snapshot_dir_pre = self.snapshot_dir
            self.snapshot_file_pre = self.snapshot_file
          
        if not os.path.exists(self.snapshot_dir +"/features_net/"):
            os.mkdir(self.snapshot_dir +"/features_net/")

        self.delete_events()

    def inputs(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("Pointnet_and_inputs"):
                self.batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
                self.bn_decay = self.get_bn_decay(self.batch, self.batch_size)
                self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
                self.alpha = tf.placeholder_with_default(0.0001, shape=None, name="alpha")
                self.Y = tf.placeholder(tf.int32, shape=[self.args.num_gpus, None], name="Y")
                self.Y_class = tf.placeholder(tf.int32, shape=[self.args.num_gpus, self.data_size], name="Y_class")
                self.Y_len = tf.placeholder(tf.int32, shape=[self.args.num_gpus, None], name="Y_len")
                # attributes for points
                self.pattr = tf.placeholder(tf.float32, shape=[self.args.num_gpus, self.data_size, self.feature_len], name="pattr")
                self.PC = tf.placeholder(tf.float32, shape=(self.args.num_gpus, self.data_size, self.n_ps, 3), name="Points")

    def tower_loss(self,is_training, bn_decay, Y, PC, Y_class, pattr, Y_len):

        Y = tf.squeeze(Y, [0])
        PC = tf.squeeze(PC, [0])
        Y_class = tf.squeeze(Y_class, [0])
        pattr = tf.squeeze(pattr, [0])
        Y_len = tf.squeeze(Y_len, [0])

        with tf.variable_scope("dgcnn"):
            net, classification_loss, pred_class = dgcnn.get_model(point_cloud=PC, labels=Y_class, is_training=is_training,
                                                                   bn_decay=bn_decay, n_classes=self.num_classes,
                                                                   class_weights=self.class_weights)

            X = tf.expand_dims(tf.cast(tf.concat([net, pattr], axis=1, name="X"), tf.float32), 0)

            pred = model.get_model(X, is_training=is_training, n_classes=self.num_classes, bn_decay=bn_decay)  # probabilities
            prediction = tf.to_int32(tf.argmax(pred, axis=1))  # hot vector

        with tf.variable_scope("losses"):
      
            Y_pred = tf.strided_slice(pred, [0, 0], [Y_len[0], self.num_classes])
            class_weights_tensor = tf.constant(self.class_weights, dtype=tf.float32)
            label_weights = tf.gather(class_weights_tensor, indices=tf.reshape(Y[:Y_len[0]], (-1,)))

            # CACLULATE LOSSES
            with tf.device("/cpu:0"):
                loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(Y[:Y_len[0]], (-1,)),
                                                              logits=tf.reshape(Y_pred, (-1, self.num_classes)),
                                                              weights=label_weights,
                                                              reduction="weighted_sum_by_nonzero_weights")

            full_loss = (loss + classification_loss)
            # loss_weight = .001
            # full_loss = (loss_weight*loss + classification_loss)

            ########################################################################################
            tf.summary.scalar('loss_classificaton', tf.cast(classification_loss, dtype=tf.float32))
            tf.summary.scalar('loss_net', loss)
            tf.summary.scalar('loss_sum', full_loss)
            #######################################################################################

            if self.args.l2_norm:
                  beta = 0.001  # weight trainable variables
                  fresh = tf.nn.l2_loss(tf.trainable_variables()[0])
                  for v in tf.trainable_variables():
                      fresh += tf.nn.l2_loss(v[:])
                  full_loss = full_loss + beta * fresh

        return full_loss, pred, prediction, classification_loss

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def make_split(self, **kwargs):
        in_splits = {}
        for k, v in kwargs.items():
            in_splits[k] = tf.split(v, self.args.num_gpus, axis=0)
        return in_splits

    def tower(self):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # Calculate the gradients for each model tower.
        tower_grads = []
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        split = self.make_split(Y=self.Y, PC=self.PC, Y_class=self.Y_class, pattr=self.pattr, Y_len=self.Y_len)
        self.prediction = {}
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d'% i) as scope:

                        self.loss, self.pred, self.prediction[i], self.classification_loss = self.tower_loss( 
                                                                                             is_training=self.is_training, bn_decay=self.bn_decay,
                                                                                             **{k : v[i] for k, v in split.items()})

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(self.loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # calculate the mean of each gradient. 
        # Note that this is the synchronization point across all towers.
        grads = self.average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        self.batch = global_step

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.bn_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        self.train = tf.group(apply_gradient_op, variables_averages_op)

    def create_saver_ops(self):
        with tf.device('/cpu:0'):
            self.saver_p = tf.train.Saver(name="saver")
            self.saver_best = tf.train.Saver(name="saver_b")
            value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dgcnn/Features_')
            self.saver_features = tf.train.Saver(value_list, max_to_keep=100)

    def build_tensors_in_checkpoint_file(self, loaded_tensors):
        full_var_list = list()

        # Loop all loaded tensors
        for i, tensor_name in enumerate(loaded_tensors[0]):
            # Extract tensor
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
                full_var_list.append(tensor_aux)
            except:
                pass

        return full_var_list

    def initialize_vars(self, session, best=False):
        if self.resume:
            if self.restoring_parameters:
                    value_list = []
                    value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

                    def initialize_uninitialized(sess):
                        global_vars = tf.global_variables()
                        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
                        # print([str(i.name) for i in not_initialized_vars])  # only for debuging
                        if len(not_initialized_vars):
                            sess.run(tf.variables_initializer(not_initialized_vars))
                    
                    ckpt = tf.train.get_checkpoint_state(self.snapshot_dir_pre + "/features_net/")
                    print('-- \t Loading Feature Model from : ', self.snapshot_dir_pre)
                    self.saver_features.restore(session, ckpt.model_checkpoint_path)
                    initialize_uninitialized(session)
                   
                    self.saver_new = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=100)
                    print("-- \t Restored")
            else:
                if self.resume_best_val:
                    ckpt = tf.train.get_checkpoint_state(self.snapshot_dir_pre+"/best_valid/")
                    if not ckpt:
                        ckpt = tf.train.get_checkpoint_state(self.snapshot_dir_pre)
                else:
                    ckpt = tf.train.get_checkpoint_state(self.snapshot_dir_pre)

                if ckpt and ckpt.model_checkpoint_path:
                    print("-- \t Restoring parameters from saved snapshot")
                    print("-- \t", self.snapshot_file_pre)
                    self.saver_p.restore(session, ckpt.model_checkpoint_path)
                    f = open(self.txt_name+".txt", "a+")
                    f.write("\n -- \t Restoring parameters from saved snapshot\n")
                    f.close()
                else:
                    print("-- \t Initializing weights to random values")
                    init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                                     resources.initialize_resources(resources.shared_resources()))
                    session.run(init_vars)

                    f = open(self.txt_name+".txt", "a+")
                    f.write("\n-- \t Initializing weights to random values\n")
                    f.close()
        else:
                print("-- \t  Initializing weights to random values")
                init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                                 resources.initialize_resources(resources.shared_resources()))
                session.run(init_vars)
                f = open(self.txt_name+".txt", "a+")
                f.write("\n-- \t Initializing weights to random values\n")
                f.close()

    def create_tensorboard_ops(self):
        with tf.variable_scope('tensorboard') as scope:
            self.merged = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_dir)#, graph=self.graph) # to save the graph
            self.writer_1 = tf.summary.FileWriter(self.tensorboard_dir + "/mean_loss_classificaton")
            self.writer_2 = tf.summary.FileWriter(self.tensorboard_dir + "/mean_loss_net")
            self.writer_3 = tf.summary.FileWriter(self.tensorboard_dir + "/mean_loss_semaseg")

    def get_bn_decay(self, batch, BATCH_SIZE ):
        BN_INIT_DECAY = 0.8
        BN_DECAY_DECAY_RATE = 0.8
        BN_DECAY_DECAY_STEP = float(300000)
        BN_DECAY_CLIP = 0.099

        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def get_learning_rate(self, batch, BATCH_SIZE):
        # This funtion can be used instead using a fixed learning rate
        BASE_LEARNING_RATE=0.01
        DECAY_RATE = 0.95
        DECAY_STEP = float(3000)
        learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,  # Decay step.
            DECAY_RATE,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.0001)  # CLIP THE LEARNING RATE!!
        return learning_rate

    def delete_events(self,tensorboard_files=False, snapshots_files=False):
        # Delete previous models
        if tensorboard_files:
            files = glob.glob(self.tensorboard_dir+"/*")
            for f in files:
                os.remove(f)
        if snapshots_files:
            files = glob.glob(self.snapshot_dir+"/*")
            for f in files:
                os.remove(f)

    def load_fist_time(self):
            self.trainlist_p_f, self.train_gt_sp_f, self.trainlist_f, self.testlist_p_f, self.test_gt_sp_f, self.testlist_f, _, _, \
            self.train_indices_list_f, self.test_indices_list_f = sp_utils.get_datasets(self.args, False)
            #print("  -- load Done --")
            return 1

    def load_again(self):
            trainlist_p_2, train_gt_sp_2, trainlist_2, testlist_p_2, test_gt_sp_2, testlist_2, train_names_2, valid_names_2, train_indices_list_2, test_indices_list_2 = sp_utils.get_datasets(self.args, False)

            self.trainlist_p = self.trainlist_p_f + trainlist_p_2
            self.train_gt_sp = self.train_gt_sp_f + train_gt_sp_2
            self.trainlist = self.trainlist_f + trainlist_2
            self.testlist_p = self.testlist_p_f + testlist_p_2
            self.test_gt_sp = self.test_gt_sp_f + test_gt_sp_2
            self.testlist = self.testlist_f + testlist_2
            self.train_indices_list = self.train_indices_list_f + train_indices_list_2
            self.test_indices_list = self.test_indices_list_f + test_indices_list_2

            #print("  -- load Done --")
            return 1

    def smart_division(self, session, xyz_points, n_sp, batch_size):
      with tf.device('/cpu:0'):
        k = 8
        pc = tf.expand_dims(tf.convert_to_tensor(xyz_points[:, :3], dtype=tf.float32), 0)
        adj = tf_util.pairwise_distance(pc)
        nn_idx = tf_util.knn(adj, k=k) 
        nn_idx = session.run(nn_idx)
        nn_idx = np.squeeze(nn_idx, 0)
        nn_idx_export= nn_idx       

        nn_idx = np.reshape(nn_idx, -1)  # create a graph using the k nearest neighbor

        batch_ind_list = []
        n_segments = []
        batch = 0
        n_segments.append(0)
        n_seg_total = 0
        cont = 0

        while (n_seg_total < n_sp):  # create a smart bacht  division based on the nearest neighboors
            n_seg = batch_size / k   # number of superpixels or segments, k number of neighboors
                                     # n_seg is the minimun number of segments with k neighboors each

            if (batch + batch_size) < len(nn_idx):
                batch_in = int(batch)
                batch_out = int(batch + batch_size)
                batch_ind = nn_idx[int(batch_in):int(batch_out)]
                unique_seg = np.shape(np.unique(batch_ind))[0]
                seg_add = int(math.floor((batch_size - unique_seg) / k))
            else:
                n_seg = int((len(nn_idx) - batch) / k)
                batch_in = int(batch)
                batch_out = len(nn_idx)
                batch_ind = nn_idx[int(batch_in):int(batch_out)]
                seg_add = 0

            while (seg_add > 0):

                if (n_seg_total + seg_add) < (n_sp):

                    if seg_add > 0:
                        batch_ind_old = np.unique(batch_ind)
                        batch_out_new = batch_out + seg_add * k
                        if batch_out_new < n_sp:

                            batch_ind_new = nn_idx[int(batch_out):int(batch_out_new)]
                            batch_ind = np.concatenate((batch_ind_old, batch_ind_new))
                            n_seg += seg_add

                            batch_out = batch_out_new

                            unique_seg = np.shape(np.unique(batch_ind))[0]
                            seg_add = int(math.floor((batch_size - unique_seg) / k))
                        else:
                            seg_add = 0
                else:
                    seg_add = 0

            n_seg_total += n_seg
            n_segments.append(int(n_seg_total))

            batch += n_seg * k
            ind = np.array(range(int(n_segments[cont]), int(n_segments[cont + 1])))
            ind = ind[ind < n_sp]
            batch_ind = np.setdiff1d(batch_ind, ind)
            batch_ind = np.concatenate((ind, batch_ind))
            rep = batch_size - np.shape(batch_ind)[0]
            try:
                batch_ind_prev = np.concatenate((batch_ind, np.random.choice(batch_ind, size=rep)))

                batch_ind_list.append(batch_ind_prev)
            except:
                seg_add = 0
                n_seg_total =2*  n_sp
            cont += 1

        n_batches = np.shape(batch_ind_list)[0]  # Num batches per epoch

        return n_batches, batch_ind_list, n_segments, nn_idx_export

    def normalization(self, xyz_points_norm, xyz_batch, att_points, batch_ind_list, batch, n_segments):

        xyz_batch = xyz_points_norm  # just xyz without rgb, already normalized
        att_points_batch = att_points[batch_ind_list[batch]]
        Y_len = np.array(n_segments[batch + 1] - n_segments[batch], dtype=np.int32)
        Y_len = np.expand_dims(Y_len, axis=1)

        return att_points_batch, Y_len, xyz_batch


    def data_prepo(self, sess, gt_aux, points_super_point_aux, sp_atrributes_aux, batch_size):

        label_del = 0 
        indices = gt_aux[:] >= label_del
        indices = np.reshape(indices, [np.shape(indices)[0]])
        gt = gt_aux[indices]
        points_super_point = points_super_point_aux[indices, :, :]  #x'y'z' xyz, rgb, z, linearity, planarity, scattering, verticality

        # sp_atrributes = sp_atrributes_aux[indices, :-1]  # xyz, length, volume, surface, number points
        sp_atrributes = sp_atrributes_aux[indices, :]  # xyz, length, volume, surface, number points
        points_number = copy.deepcopy(sp_atrributes_aux[indices, -1])
        points_number_normalized = np.log(points_number/40.)
        print("unique GT in training : ", np.unique(gt))
      
        del gt_aux, points_super_point_aux, sp_atrributes_aux
        rgb = self.rgb  # 6 new experiment, 3 old experiment
        xyz_points = points_super_point[:, :, :3]  # x´y´z´ , not rgb set of points inside segment
        rgb_points = points_super_point[:, :, 6:9]  
        att_points = points_super_point[:, :, 3+6:]  # (not xyz, rgb), consider 5 attributes  6:11
       
        if rgb > 3:
            rgb_points = rgb_points / 255.0 - 0.5
            xyz_points = np.concatenate((xyz_points, rgb_points), axis=1)

        min_z = np.expand_dims(np.min(att_points[:, :, 0], axis=1), axis=1)  # att_points 0 : Z'
        max_z = np.expand_dims(np.max(att_points[:, :, 0], axis=1), axis=1)
        min_z[min_z < 0] = 0.     #  minimum value for z is 0 
        min_z[min_z > 20] = 20.   #  maximum value for z is 20 
        min_z = (min_z) / (20.)   #  values normalized from 0 to 20

        max_z[max_z < 0] = 0.     #  minimum value for z is 0 
        max_z[max_z > 20] = 20.   #  maximun value for z is 20 
        max_z = (max_z) / (20.)   #  values normalized from 0 to 20
        
        att_points = np.mean(att_points, axis=1)  # store mean of: z, linearity, planarity, scattering, verticality
        indices_a = att_points[:, 0] < 0.  #  minimum value for z is 0 
        att_points[indices_a, 0] = 0.      #  minimum value for z is 0 
        indices_a = att_points[:, 0] > 20. #  maximun value for z is 20 
        att_points[indices_a, 0] = 20.     #  maximun value for z is 20 
        att_points[:, 0] = (att_points[:, 0]) / 20. #  values normalized from 0 to 20

        sp_atrributes[:, -1] = points_number_normalized
        att_points = np.concatenate((min_z, max_z, att_points, sp_atrributes[:, 3:]), axis=1)

        ################################################################################################################
        #  include mean xyz
        ################################################################################################################
        xyz_points_aux = points_super_point[:, :, 3:6]  # xyz all point in the original coordinates system
        xyz_points_aux = np.mean(xyz_points_aux, axis=1)
        att_points = np.concatenate((xyz_points_aux, att_points), axis=1)

        ################################################################################################################
        # store mean xyz min z, max z, mean of: z, linearity, planarity, scattering, verticality
        # xyz centroid, length, volume, surface, number of points of segment.
        # xyz centroind for smart division after that is not more necessary
        ################################################################################################################
        n_sp = np.shape(gt)[0]

        xyz_points_centroids = xyz_points_aux
        # print("to smart division") 
        n_batches, batch_ind_list, n_segments, nn_idx = self.smart_division(sess, xyz_points_centroids, n_sp, batch_size)
        # print("Done smart division")
     
        return n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point, points_number

    def get_batch(self, batch, xyz_points, gt, att_points, points_super_point, batch_ind_list, n_segments):

        gt_batch = np.reshape(gt[batch_ind_list[batch]], (self.batch_size))
        xyz_batch = xyz_points[batch_ind_list[batch]]
        xyz_points_norm = points_super_point[:, :, :3]  # xyz norm
        xyz_points_norm = xyz_points_norm[batch_ind_list[batch]]
        att_points_batch, Y_len, xyz_batch = self.normalization(xyz_points_norm, xyz_batch, att_points,
                                                                batch_ind_list, batch, n_segments)
        Y = np.zeros(np.shape(gt_batch))
        Y[: Y_len[0]] = gt_batch[:Y_len[0]]  # padding
        # att_points_batch[att_points_batch < 0] = 0.   # inferior limit 
        # att_points_batch[att_points_batch > 30] = 30. # superior limit

        return xyz_batch, att_points_batch, Y, gt_batch, Y_len

    def pool_train(self, sess, epoch):

        batch_size = self.batch_size
        array_samples = np.arange(self.n_samples)
        np.random.shuffle(array_samples)
        total_mean_loss = 0.0
        total_mean_class_loss = 0.0

        for sample in array_samples:
            print("\nSample {}/{}".format(sample + 1, self.n_samples))  # Number of Training scenes
            total_Acc = 0.0
            total_loss_batch = 0.0

            points_super_point_aux = self.trainlist_p[sample]
            gt_aux = self.train_gt[sample]
            sp_atrributes_aux = self.trainlist[sample]

            n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point, _ = self.data_prepo(
                sess, gt_aux, points_super_point_aux, sp_atrributes_aux, batch_size)

            del points_super_point_aux, gt_aux, sp_atrributes_aux

            # for batch in tqdm(range(0, int(n_batches/self.args.num_gpus), self.args.num_gpus)):
            for batch in (range(0, int(n_batches / self.args.num_gpus), self.args.num_gpus)):
                
                xyz_batch_l = []
                att_points_batch_l = []
                Y_l = []
                Y_class_l = []
                Y_len_l = []

                for i in range(self.args.num_gpus):
                    xyz_batch, att_points_batch, Y, Y_class, Y_len = self.get_batch(batch + i, xyz_points, gt,
                                                                                    att_points, points_super_point,
                                                                                    batch_ind_list, n_segments)

                    xyz_batch_l.append(xyz_batch)
                    att_points_batch_l.append(att_points_batch)
                    Y_l.append(Y)
                    Y_class_l.append(Y_class)
                    Y_len_l.append(Y_len)

                feed_dict = {self.PC: xyz_batch_l, self.pattr: att_points_batch_l, self.Y: Y_l,
                             self.Y_class: Y_class_l, self.Y_len: Y_len_l, self.is_training: True}

                _, l, acc, summary = sess.run([self.train, self.loss, self.classification_loss, self.merged],
                                              feed_dict=feed_dict)
                total_loss_batch += l
                total_Acc += acc

                del xyz_batch, att_points_batch, Y, Y_class, Y_len
                del xyz_batch_l, att_points_batch_l, Y_l, Y_class_l, Y_len_l
                self.summary_writer.add_summary(summary) 

            total_mean_loss += total_loss_batch / int(n_batches / self.args.num_gpus)
            total_mean_class_loss += total_Acc / int(n_batches / self.args.num_gpus)

            print("Mean loss: %f, Mean class loss: %f, n_sp: %d" % (
                total_loss_batch / int(n_batches / self.args.num_gpus),
                total_Acc / int(n_batches / self.args.num_gpus), n_sp))

            f = open(self.txt_name + ".txt", "a+")
            f.write("\n Epoch: %i, Sample:                %s \n" % (epoch, sample))
            f.write("Mean loss: %f, Mean class loss: %f, n_sp: %d" % (
            total_loss_batch / int(n_batches / self.args.num_gpus),
            total_Acc / int(n_batches / self.args.num_gpus), n_sp))

            f.close()

            if math.isnan(total_loss_batch / int(n_batches / self.args.num_gpus)):
                import sys
                sys.exit()


            del n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point

      
        summary = tf.Summary(
            value=[tf.Summary.Value(tag="mean_loss", simple_value=total_mean_class_loss / self.n_samples), ])
        self.writer_1.add_summary(summary)  # mean_loss_classificaton
        self.writer_1.flush()

        summary2 = tf.Summary(
            value=[tf.Summary.Value(tag="mean_loss", simple_value=total_mean_loss / self.n_samples), ])
        self.writer_2.add_summary(summary2)  # mean_loss_net
        self.writer_2.flush()

        summary3 = tf.Summary(
            value=[tf.Summary.Value(tag="mean_loss", simple_value=((total_mean_loss -total_mean_class_loss) / self.n_samples)), ])
        self.writer_3.add_summary(summary3)  # mean_loss_net - mean_loss_classificaton
        self.writer_3.flush()

        # Save snapshot
        k = 0

        while k < 2:

            try:
                self.saver_p.save(sess, self.snapshot_file, global_step=epoch)
                print("     saved!")
                self.saver_features.save(sess, self.snapshot_dir_pre + "/features_net/" + "snapshot.chk")
                k = 2
            except:
                k += 1
                print("oopss! try again")
                # print("-- Train Done --")
            
    def final_process(self, train_, load_):

        self.trainlist_p = load_["trainlist_p"]
        self.train_gt_sp = load_["train_gt_sp"]
        self.trainlist_p = load_["trainlist_p"]
        self.train_gt_sp = load_["train_gt_sp"]
        self.trainlist =  load_["trainlist"]
        self.testlist_p= load_["testlist_p"]
        self.test_gt = load_["test_gt_sp"]
        self.testlist = load_["testlist"]
        self.train_indices_list = load_["train_indices_list"]
        self.test_indices_list= load_["test_indices_list"]

        print("Done", train_)

        return train_
 
    def validation(self, sess, epoch, best_AIoU):
       batch_size = self.batch_size
       
       for sample in range(self.n_samples_testing):
                    print("\nSample validation {}/{}".format(sample+1, self.n_samples_testing))  # Number of Training scenes

                    total_loss_batch = 0.0
                    total_Acc = 0.0


                    points_super_point_aux = self.testlist_p[sample]
                    gt_aux = self.test_gt[sample]
                    sp_atrributes_aux = self.testlist[sample]

                    n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point, _ = self.data_prepo(
                        sess, gt_aux, points_super_point_aux, sp_atrributes_aux, batch_size)

                    sample_prediction = np.zeros(n_sp, dtype=np.int32)

                    # for batch in tqdm(range(0, int(n_batches/self.args.num_gpus), self.args.num_gpus)):
                    for batch in (range(0, int(n_batches / self.args.num_gpus), self.args.num_gpus)):

                        xyz_batch_l = []
                        att_points_batch_l = []
                        Y_l = []
                        Y_class_l = []
                        Y_len_l = []

                        for i in range(self.args.num_gpus):

                            xyz_batch, att_points_batch, Y, Y_class, Y_len = self.get_batch(batch+i, xyz_points, gt,
                                                                                        att_points, points_super_point,
                                                                                        batch_ind_list, n_segments)

                            xyz_batch_l.append(xyz_batch)
                            att_points_batch_l.append(att_points_batch)
                            Y_l.append(Y)
                            Y_class_l.append(Y_class)
                            Y_len_l.append(Y_len)

                        feed_dict = {self.PC: xyz_batch_l, self.pattr: att_points_batch_l, self.Y: Y_l,
                                     self.Y_class: Y_class_l, self.Y_len: Y_len_l, self.is_training: False}

                        l, acc, _= sess.run([self.loss, self.classification_loss, self.pred], feed_dict=feed_dict)
                        total_loss_batch += l
                        total_Acc += acc

                        pred = sess.run([self.prediction], feed_dict=feed_dict)
                        pred_ext = []

                        for i in range(self.args.num_gpus):
                            pred_ext.extend(pred[0][i])

                        pred_ext = np.reshape(pred_ext, self.batch_size*self.args.num_gpus)
                        sample_prediction[n_segments[batch]:n_segments[batch + self.args.num_gpus]] =\
                            pred_ext[:(n_segments[batch + self.args.num_gpus] - n_segments[batch])]

                        del xyz_batch, att_points_batch, Y, Y_class, Y_len
                        del xyz_batch_l, att_points_batch_l, Y_l, Y_class_l, Y_len_l

                    print("Mean loss: %f, Mean class loss: %f, n_sp: %d" % (
                        total_loss_batch / int(n_batches / self.args.num_gpus),
                        total_Acc / int(n_batches / self.args.num_gpus), n_sp))

                    gt_sample = np.reshape(gt, n_sp)
                    metric_dataset = metric.ConfusionMatrix(self.num_classes)
                    metric_dataset.count_predicted_batch(ground_truth_vec=gt_sample, predicted=sample_prediction)
                    cm = np.array(metric_dataset.get_confusion_matrix())
                    np.set_printoptions(suppress=True, precision=3)
                    # print("confusion matrix: ", cm)
                    recall = 100* np.diag(cm) / (np.sum(cm, axis=1)+.00001)
                    precision = 100 * np.diag(cm) / (np.sum(cm, axis=0)+.00001)
                    F1 = 2 * (precision*recall)/ (precision+recall+.000001)
                    mean_F1 = np.mean(F1)
                    print("\t Classes in GT : ",  np.unique(gt_sample))
                    print("\t Classes Output: ", np.unique(gt_sample))
                    print("\t OA : %.3f \t AIoU : %.3f \t mean F1 : %.3f " % (
                          metric_dataset.get_overall_accuracy(), metric_dataset.get_average_intersection_union(), mean_F1))
                    
                    print("\t IoU per class : ", ["%.3f " % e for e in metric_dataset.get_intersection_union_per_class()])

                    print("\t Recall    : ", recall)
                    print("\t Precision : ", precision)
                    print("\t F1-score  : \t", F1)
                  
                    f = open(self.txt_name + ".txt", "a+")
                    f.write("\n Epoch: %i, Testing Sample: %s \n" % (epoch, sample))
                    f.write("Mean loss for validation: %.4f , n_sp: %d" % (
                             total_loss_batch / int(n_batches / self.args.num_gpus), n_sp))
                    f.write("OA   : %.4f" % metric_dataset.get_overall_accuracy())
                    f.write("AIoU : %.4f" % metric_dataset.get_average_intersection_union())
                    f.close()

                    AIoU = metric_dataset.get_average_intersection_union()
                    
                    del n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point

                    if best_AIoU<AIoU:
                        best_AIoU = AIoU
                        # Save snapshot of the best validation
                        k = 0
                        if not os.path.isdir(self.snapshot_dir+"/best_valid"):
                            os.mkdir(self.snapshot_dir+"/best_valid")
                       
                        best_snapshot_file = self.snapshot_dir+"/best_valid/"  +str(epoch) + "snapshot.chk"
                        while k < 2:
                            try:
                                self.saver_best.save(sess, best_snapshot_file)
                                print("-- \t     saved Best!")
                                k = 2
                            except:
                                k += 1
                                print("-- \t error! trying again")

                        f = open(self.txt_name + "_best_validation_"+ ".txt", "a+")
                        f.write("\n#######################################################################"
                                "\n                               Epoch : %i                              "
                                "\n#######################################################################" % epoch)
                        f.write("\n Epoch: %i, Sample:%s \n" % (epoch, sample))
                        f.write("AO: %f, AIoU: %f, n_sp: %d" % (metric_dataset.get_overall_accuracy(), AIoU, n_sp))

                        f.close()
                    return AIoU


    def train(self, n_epochs, learning_rate):

      self.learning_rate = learning_rate
      with self.graph.as_default():
        self.inputs()
        self.tower()
        self.create_tensorboard_ops()
        self.create_saver_ops()
        self.load_fist_time()
        if self.args.gpu_memory_fraction ==-1:
           config = tf.ConfigProto(allow_soft_placement=True)
        else:
           config=tf.ConfigProto(gpu_options={'allow_growth': True},allow_soft_placement=True)
           config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_memory_fraction
        
        with tf.Session(graph=self.graph, config=config) as sess:

            self.initialize_vars(sess)
            # Run the initializer
            print("-- \t Number of learnable parameters", 
                  np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            best_AIoU = 0
            # Training
            for epoch in range(n_epochs+1):

                print("\nEPOCH {}/{}".format(epoch+1, n_epochs))

                if not epoch % self.args.freq_aug and epoch != 0:
                      self.load_again()
                
                f = open(self.txt_name + ".txt", "a+")
                f.write("\n#######################################################################"
                        "\n                               Epoch : %i                              "
                        "\n#######################################################################" % epoch)
                f.close()
                self.pool_train(sess, epoch)

                # #####################################################################################################
                #                                                                                 Testing in validation
                # #####################################################################################################
                
                if epoch % self.args.freq_validation == 0 and epoch != 0:
                   AIoU = self.validation(sess, epoch, best_AIoU)
                   if AIoU > best_AIoU:
                      best_AIoU = AIoU 
                                    

    def testing(self, train_names, valid_names, class_weights, best=False):

        with self.graph.as_default():
          self.inputs()
          self.tower()
          self.create_tensorboard_ops()
          self.create_saver_ops()

          your_path = self.args.SEMA3D_PATH
          path_list = your_path.split(os.sep)
          file_name = path_list[len(path_list)-2]
         
          out_dir = self.args.SEMA3D_PATH + "/results"
         
          if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         
          if self.args.gpu_memory_fraction ==-1:
           config = tf.ConfigProto(allow_soft_placement=True)
          else:
           config=tf.ConfigProto(gpu_options={'allow_growth': True},allow_soft_placement=True)
           config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_memory_fraction
        
          with tf.Session(graph=self.graph, config=config) as sess:

            self.initialize_vars(sess, best)
            # Run the initializer
            print("Number of parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            batch_size = self.batch_size
            ############################################################################################################
            #                                                                                    Saving Training Results
            ############################################################################################################
            self.batch_size = self.data_size
            if not self.args.only_test:
             for sample in range(self.n_samples):
                print("\nSaving train Sample {}/{}".format(sample + 1, self.n_samples))  # Number of Training scenes

                points_super_point_aux = self.trainlist_p[sample]
                gt_aux = self.train_gt[sample]
                sp_atrributes_aux = self.trainlist[sample]
                gt_aux[gt_aux < 0] = self.num_classes

                n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point,_ = self.data_prepo(
                    sess, gt_aux, points_super_point_aux, sp_atrributes_aux, batch_size)

                #  edges = self.get_edges()

                del points_super_point_aux, gt_aux, sp_atrributes_aux
                #  total_loss_batch = 0.0
                sample_prediction = np.zeros(n_sp, dtype=np.int32)
                rw = np.shape(self.train_indices_list[sample])
                unary_sample = np.zeros((n_sp, self.num_classes + 3), dtype=np.float32)
                unary_sample_crf = np.zeros((n_sp, self.num_classes), dtype=np.float32)
                unary_prediction_crf = 0.125 * np.ones((int(rw[0]), self.num_classes), dtype=np.float32)
                unary_prediction = np.zeros((int(rw[0]), self.num_classes + 3), dtype=np.float32)
                #  att_Y_F  = np.zeros((0, 12))
                for batch in (range(0, int(n_batches / self.args.num_gpus), self.args.num_gpus)):  # tqdm

                    xyz_batch_l = []
                    att_points_batch_l = []
                    Y_l = []
                    Y_class_l = []
                    Y_len_l = []

                    for i in range(self.args.num_gpus):
                        xyz_batch, att_points_batch, Y, Y_class, Y_len = self.get_batch(batch + i, xyz_points, gt,
                                                                                        att_points, points_super_point,
                                                                                        batch_ind_list, n_segments)

                        xyz_batch_l.append(xyz_batch)
                        att_points_batch_l.append(att_points_batch)
                        Y_l.append(Y)
                        Y_class_l.append(Y_class)
                        Y_len_l.append(Y_len)

                    feed_dict = {self.PC: xyz_batch_l, self.pattr: att_points_batch_l, self.Y: Y_l,
                                 self.Y_class: Y_class_l, self.Y_len: Y_len_l, self.is_training: False}

                    unary = sess.run([self.pred], feed_dict=feed_dict)
                    pred = sess.run([self.prediction], feed_dict=feed_dict)
                    pred_ext = []
                    unary_ext = []

                    for i in range(self.args.num_gpus):
                        pred_ext.extend(pred[0][i])
                    unary_ext.extend(unary[0])

                    pred_ext = np.reshape(pred_ext, self.batch_size * self.args.num_gpus)

                    sample_prediction[n_segments[batch]:n_segments[batch + self.args.num_gpus]] = \
                        pred_ext[:(n_segments[batch + self.args.num_gpus] - n_segments[batch])]

                    unary_ext = np.reshape(unary_ext, (self.batch_size * self.args.num_gpus, self.num_classes))
                    unary_ext_crf = np.copy(unary_ext)
                    # correct solution:

                    def softmax(z):
                        assert len(z.shape) == 2
                        s = np.max(z, axis=1)
                        s = s[:, np.newaxis]  # necessary step to do broadcasting
                        e_x = np.exp(z - s)
                        div = np.sum(e_x, axis=1)
                        div = div[:, np.newaxis]  # dito
                        return e_x / div

                    unary_ext = softmax(unary_ext)
                  
                    att_points_batch_l = np.squeeze(att_points_batch_l, 0)
                    unary_ext = np.hstack((unary_ext, att_points_batch_l[:, 3:3+3]))

                    unary_sample[n_segments[batch]:n_segments[batch + self.args.num_gpus], :] = \
                        unary_ext[:(n_segments[batch + self.args.num_gpus] - n_segments[batch]), :]

                    unary_sample_crf[n_segments[batch]:n_segments[batch + self.args.num_gpus], :] = \
                        unary_ext_crf[:(n_segments[batch + self.args.num_gpus] - n_segments[batch]), :]

                    # ---------------------------------------------------------------------------------------------------

                    # --------------------------------------------------------------------------------------------------

                    del xyz_batch_l, att_points_batch_l, Y_l, Y_class_l, Y_len_l
             
                gt_sample = np.reshape(gt, n_sp)
                metric_dataset = metric.ConfusionMatrix(self.num_classes + 1)
                metric_dataset.count_predicted_batch(ground_truth_vec=gt_sample, predicted=sample_prediction)
                print("OA : %.4f" % metric_dataset.get_overall_accuracy())
                print("AIoU : %.4f" % metric_dataset.get_average_intersection_union())
                print("IoU per class : ", ["%.3f " % e for e in metric_dataset.get_intersection_union_per_class()])
                print("classes: ", np.unique(gt_sample))

                sample_prediction_full = np.zeros(np.shape(self.train_indices_list[sample]))
                sample_prediction_full[self.train_indices_list[sample] == 1] = sample_prediction + 1
                sample_prediction = sample_prediction_full

                unary_prediction[self.train_indices_list[sample] == 1] = unary_sample
                unary_prediction_crf[self.train_indices_list[sample] == 1] = unary_sample_crf
    

                with h5py.File(os.path.join(out_dir, 'predictions_' + train_names[sample] + '.h5'), 'w') as hf:

                    hf.create_dataset(name=train_names[sample], data=sample_prediction)  # (0-based classes)
                    print(" -- prediction save ", train_names[sample])

                with h5py.File(os.path.join(out_dir, 'unary_predictions_' + train_names[sample] + '.h5'), 'w') as hf:

                    hf.create_dataset(name=train_names[sample], data=unary_prediction)  # (0-based classes)
                    print("-- unary save ", train_names[sample])

            ############################################################################################################
            #                                                                                    Saving Testing Results
            ############################################################################################################

            for sample in range(self.n_samples_testing):
                tic()
                print("\nSaving testing Sample {}/{}".format(sample + 1,
                                                             self.n_samples_testing))  # Number of Training scenes
                points_super_point_aux = self.testlist_p[sample]
                gt_aux = self.test_gt[sample]
                sp_atrributes_aux = self.testlist[sample]
                gt_aux[gt_aux <= 0] = self.num_classes

                n_sp, n_batches, batch_ind_list, n_segments, xyz_points, gt, att_points, points_super_point, points_number = self.data_prepo(
                    sess, gt_aux, points_super_point_aux, sp_atrributes_aux, batch_size)

                sample_prediction = np.zeros(n_sp, dtype=np.int32)

                rw = np.shape(self.test_indices_list[sample])
                unary_sample = np.zeros((n_sp, self.num_classes+3), dtype=np.float32)
                unary_prediction = np.zeros((int(rw[0]), self.num_classes+3), dtype=np.float32)
                points_number_sample = np.zeros(int(rw[0]), dtype=np.int32)
                print("number of sp", n_sp)
                for batch in (range(0, int(n_batches / self.args.num_gpus), self.args.num_gpus)):  # tqdm
                    xyz_batch_l = []
                    att_points_batch_l = []
                    Y_l = []
                    Y_class_l = []
                    Y_len_l = []

                    for i in range(self.args.num_gpus):
                        xyz_batch, att_points_batch, Y, Y_class, Y_len = self.get_batch(batch + i, xyz_points, gt,
                                                                                        att_points, points_super_point,
                                                                                        batch_ind_list, n_segments)

                        xyz_batch_l.append(xyz_batch)
                        att_points_batch_l.append(att_points_batch)
                        Y_l.append(Y)
                        Y_class_l.append(Y_class)
                        Y_len_l.append(Y_len)

                    feed_dict = {self.PC: xyz_batch_l, self.pattr: att_points_batch_l, self.Y: Y_l,
                                 self.Y_class: Y_class_l, self.Y_len: Y_len_l, self.is_training: False}

                    unary = sess.run([self.pred], feed_dict=feed_dict)
                    pred = sess.run([self.prediction], feed_dict=feed_dict)
                    pred_ext = []
                    unary_ext = []

                    for i in range(self.args.num_gpus):
                        pred_ext.extend(pred[0][i])
                    unary_ext.extend(unary[0])

                    pred_ext = np.reshape(pred_ext, self.batch_size * self.args.num_gpus)
                    sample_prediction[n_segments[batch]:n_segments[batch + self.args.num_gpus]] = \
                        pred_ext[:(n_segments[batch + self.args.num_gpus] - n_segments[batch])]

                    unary_ext = np.reshape(unary_ext, (self.batch_size * self.args.num_gpus, self.num_classes))

                    # correct solution:

                    def softmax(z):
                        assert len(z.shape) == 2
                        s = np.max(z, axis=1)
                        s = s[:, np.newaxis]  # necessary step to do broadcasting
                        e_x = np.exp(z - s)
                        div = np.sum(e_x, axis=1)
                        div = div[:, np.newaxis]  # dito
                        return e_x / div
                    
                    unary_ext = softmax(unary_ext)
                    att_points_batch_l = np.squeeze(att_points_batch_l, 0)
                    unary_ext = np.hstack((unary_ext, att_points_batch_l[:, 3:3+3]))
                    unary_sample[n_segments[batch]:n_segments[batch + self.args.num_gpus], :] = \
                        unary_ext[:(n_segments[batch + self.args.num_gpus] - n_segments[batch]), :]

                    del xyz_batch_l, att_points_batch_l, Y_l, Y_class_l, Y_len_l

                sample_prediction_full = np.zeros(np.shape(self.test_indices_list[sample]))
                sample_prediction_full[self.test_indices_list[sample] == 1] = sample_prediction + 1
                sample_prediction = sample_prediction_full
                unary_prediction[self.test_indices_list[sample] == 1] = unary_sample

                print(np.unique(sample_prediction))
               
                toc()
                with h5py.File(os.path.join(out_dir, 'predictions_' + valid_names[sample] + '.h5'), 'w') as hf:

                    hf.create_dataset(name=valid_names[sample], data=sample_prediction)  # (0-based classes)
                   
                with h5py.File(os.path.join(out_dir, 'unary_predictions_' + valid_names[sample] + '.h5'), 'w') as hf:

                    hf.create_dataset(name=valid_names[sample], data=unary_prediction)  # (0-based classes)
                    print(valid_names[sample])
                print("-- Done")
    


