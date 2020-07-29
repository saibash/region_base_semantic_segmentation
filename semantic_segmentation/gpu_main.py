#from tqdm import tqdm
import argparse
import os.path as path
import sys
import numpy as np
import tensorflow as tf
from model import sp_utils
from model import train_batch
from model import data_processing as dp

two_up =  path.abspath(path.join(__file__ ,"../../.."))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser_att():
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation with Edge Convolutions')
    
    # Dataset
    parser.add_argument('--dataset', default='other', help='Dataset name: sema3d|s3dis')
    parser.add_argument('--odir', default='log', help='Output directory to store logs')
    parser.add_argument('--log_file', default='log_doc', help='Log file to store results')
    parser.add_argument('--resume', default=True, type=str2bool, 
                        help='True: load a previously saved model')
    parser.add_argument('--resume_best_val',  default=False, type=str2bool, 
                        help='True: load model from best validation result')
    parser.add_argument('--restoring_partially',  default=False, type=str2bool, 
                        help='True: to initiallize with pretraining model')
    parser.add_argument('--db_train_name', default='training', help='Training folder')
    parser.add_argument('--db_test_name',  default='testing', help='Testing or validation folder') 
    parser.add_argument('--SEMA3D_PATH',   default='/Data/forest4D_dlr', help='Dataset directory') 
    parser.add_argument('--pre_train',  default='dlr_old_model', help="pretrain model to initialize new model")
    parser.add_argument('--model_name', default='dlr_new_model', help="model_name")

    # Point cloud pre-processing
    parser.add_argument('--pc_xyznormalize', default=True, type=str2bool,
                        help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')
    parser.add_argument('--pc_augm', default=False, type=float, help='Training augmentation')
    parser.add_argument('--pc_augm_scale', default=1.1, type=float,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=True, type=str2bool,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float,
                        help='Training augmentation: Probability of mirroring about x or y axes') 
    parser.add_argument('--pc_augm_jitter', default=0, type=str2bool,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--spg_attribs01', default=1, type=str2bool,
                        help='Bool, normalize edge features to 0 mean 1 deviation')

    # Model configuration
    parser.add_argument('--n_classes',  default=3, type=int, help='Number of classes')
    parser.add_argument('--ptn_minpts', default=40, type=int,
                        help='Minimum number of points into a segment for computing its embedding') 
    parser.add_argument('--ptn_npts', default=512, type=int, 
                        help='Maximum number of points into a segment for computing its embedding')  
    parser.add_argument('--rgb', default=False, type=str2bool, help="Consider RGB on Training") 
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate for training')
    parser.add_argument('--class_weights', default=False, type=str2bool, help='Compute class weights for imbalanced datasets')
    parser.add_argument('--sor', default=False, type=str2bool,    help='Statistical Outlier Removal')
    parser.add_argument('--mean_k', default=30, type=float,   help='mean_k for Statistical Outlier Removal')
    parser.add_argument('--std_dev', default=1.0, type=float, help='std_dev for Statistical Outlier Removal')
    parser.add_argument('--l2_norm', default=True, type=float, help='l2 normalization') 
    
    # Computation configuration
    parser.add_argument('--training', default=True, type=str2bool, help='True:Training, False:Testing')
    parser.add_argument('--only_test', default=True, type=str2bool, help='True:Evaluate test, False:evaluate training and testing')
    parser.add_argument('--batch_size', default=1024, type=float, help='Batch size for training')
    parser.add_argument('--n_epochs', default=1000, type=float,   help='Number of epochs')
    parser.add_argument('--freq_aug', default=5, type=int,  
                        help='Frequency in epochs of training augmentation')
    parser.add_argument('--freq_validation', default=5, type=int, help='Frequency validation in epochs')
    parser.add_argument('--num_gpus', default=1, type=int, help='int, How many GPUs to use.')
    parser.add_argument('--gpu_memory_fraction', default=.5, type=float,
                       help='Fraction of the GPU to be allocated to that process, -1 to choose automatically')
    
    
    args = parser.parse_args()
    return args

def train(args):
        print("-- \t training...")
        trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist, train_names, valid_names, train_indices_list, test_indices_list = sp_utils.get_datasets(args, farthest=False)

        trainlist_p_2, train_gt_sp_2, trainlist_2, testlist_p_2, test_gt_sp_2, testlist_2, train_names_2, valid_names_2, train_indices_list_2, test_indices_list_2 = sp_utils.get_datasets(args, farthest=False)

        trainlist_p =  trainlist_p + trainlist_p_2
        train_gt_sp = train_gt_sp + train_gt_sp_2
        trainlist = trainlist + trainlist_2
        testlist_p = testlist_p + testlist_p_2
        test_gt_sp = test_gt_sp + test_gt_sp_2
        testlist = testlist + testlist_2
        # train_names = train_names + train_names_2
        # valid_names = valid_names + valid_names_2
        train_indices_list = train_indices_list + train_indices_list_2
        test_indices_list = test_indices_list + test_indices_list_2

# #        # print("\n \n number of training scenes = ", np.shape(trainlist_p)[0])
# #        # print("number of super points in first scene = ", np.shape(train_gt_sp[0])[0])
# #        # print("number of testing scenes = ", np.shape(testlist_p)[0])
# #        # print("number of super points in first scene = ", np.shape(test_gt_sp[0])[0])
         
        if args.class_weights:
             class_weights = dp.calculate_class_weights(train_gt_sp, n_classes=args.n_classes, method="paszke", c=1.10)
             print("Class weights : ",class_weights)
        else:
             class_weights = np.ones((args.n_classes,))
  
        
#
        model = train_batch.SuperPointNet(trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist,
                                          train_indices_list, test_indices_list, args=args,
                                          class_weights=class_weights) 

# #         # model.delete_events(tensorboard_files=True, snapshots_files=True)
# #         # model.restoring_parameters = True

        model.train(n_epochs=args.n_epochs, learning_rate=args.learning_rate)


def test(args):
     print("-- \t testing...")
     trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist, train_names, valid_names, train_indices_list, test_indices_list = sp_utils.get_datasets(args, False)
     class_weights = np.ones((args.n_classes,))
     model = train_batch.SuperPointNet(trainlist_p, train_gt_sp, trainlist, testlist_p, test_gt_sp, testlist,
                                  train_indices_list, test_indices_list, args=args, class_weights=class_weights)                       

     model.testing(train_names=train_names, valid_names=valid_names, class_weights=class_weights)

def main():
    args = parser_att()
    if args.training:
       train(args)
    else:
       test(args)

if __name__ == "__main__":
    main()


