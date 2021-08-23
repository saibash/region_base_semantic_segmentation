# Region-Based Edge Convolutions With Geometric Attributes for the Semantic Segmentation of Large-Scale 3-D Point Clouds 

This is an implementation of the paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9103287

by : Jhonatan Contreras 

## Code structure

###  ./segmentation_and_rebuild 
  - ./partition/segmentation.py - geometric partitioning and segments construction using handcrafted features.
  - ./partition/write_segments_to_pc.py - upsample the prediction to the original point clouds..
  
###  ./semantic_segmentation
 - ./gpu_main.py -  supervised learning code semantic segmentation and inference.
 

## Disclaimer

The partition method is stochastic. The results obtained could differ slightly. The original partition work was developed for Loic Landrieu, and presented in the work: 

<i>A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds. Loic Landrieu, Hugo Raguet , Bruno Vallet , Clément Mallet, Martin Weinmann</i>


## Requirements
    
1. Download current version of the repository.
2. Follow the document: <b> installation.ipynb </b> 
    

## Datasets and visualization

In order to use our code, the dataset must be divided into diferent subfolders, e.g. <b>testing/, training/, validation/</b>.

The input files ".txt" and ".npy" are supported. It is recommended that other formats are first transformed to ".txt" using a third party software, e.g. CloudCompare &copy;

CloudCompare &copy; could be used to visualize intermediate and final results.
To install CloudCompare &copy; use the command line:

    $ sudo snap install cloudcompare



## Running the code

To run our code from scratch on different datasets, we need to complete four stages:

 <ol>
  <li>Segmentation</li>
  <li>Training</li>
  <li>Inference</li>
  <li>Upsampling</li>
</ol>  

### Segmentation 

We recomend to follow the file <b> ./segmentation_and_rebuild/main.sh </b> as guide using the sample dataset <b><i>forest4D_dlr</t></b>.

The code <b>./partition/segmentation.py </b> has the following arguments:
    
#### Required arguments.

 <table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--path_to_data</td>
    <td>"../Data/forest4D_dlr"</td>
    <td>str</td>
    <td>Path to data</td>
  </tr>
  <tr>
    <td>--path_to_output</td>
    <td>"../Data/forest4D_dlr"</td>
    <td>str</td>
    <td>Path to output</td>
  </tr>
  <tr>
    <td>--n_labels</td>
    <td>4</td>
    <td>int</td>
    <td>number of classes</td>
  </tr>
  <tr>
    <td>--areas</td>
    <td>"testing/, training/, validation/"</td>
    <td>str</td>
    <td>"list of subfolders to be processed separated by ( , )</td>
  </tr> 
</table> 


#### Optional arguments.
  <table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--RGB</td>
    <td>"False"</td>
    <td>bool</td>
    <td>True if Data set contains RGB information</td>
  </tr>
  <tr>
    <td>--version</td>
    <td>"V0"</td>
    <td>str</td>
    <td>for multiples segmentation parameters, use a different output name </td>
  </tr>
  <tr>
    <td>--file_extension</td>
    <td>".txt"</td>
    <td>str</td>
    <td>file extension</td>
  </tr>
  <tr>
    <td>--gt_index</td>
    <td>3</td>
    <td>int</td>
    <td>ground true index in file</td>
  </tr>
  <tr>
    <td>--rgb_intensity_index</td>
    <td>[3,4,5]</td>
    <td>list</td>
    <td>rgb or intensity index in file</td>
  </tr>
  <tr>
    <td>--ver_batch</td>
    <td>2000000</td>
    <td>int</td>
    <td>Batch size for reading large files</td>
  </tr>
  <tr>
    <td>--voxel_width</td>
    <td>.01</td>
    <td>float</td>
    <td>voxel size when subsampling (in m)</td>
  </tr>
    <tr>
    <td>--k_nn_geof</td>
    <td>45</td>
    <td>int</td>
    <td>number of neighbors for the geometric features</td>
  </tr>
    <tr>
    <td>--k_nn_adj</td>
    <td>10</td>
    <td>int</td>
    <td>adjacency structure for the minimal partition</td>
  </tr>
    <tr>
    <td>--lambda_edge_weight</td>
    <td>1.</td>
    <td>float</td>
    <td>parameter determine the edge weight for minimal part.</td>
  </tr>
   <tr>
    <td>--reg_strength</td>
    <td>1.</td>
    <td>float</td>
    <td>regularization strength for the minimal partition</td>
  </tr>
   <tr>
    <td>--sorted</td>
    <td>False</td>
    <td>bool</td>
    <td>Reverse order to read the data</td>
  </tr>
   <tr>
    <td>--overwrite</td>
    <td>False</td>
    <td>bool</td>
    <td>Consider previous results</td>
  </tr>
  <tr>
    <td>--print_progress</td>
    <td>True</td>
    <td>bool</td>
    <td>Print current process</td>
  </tr>
</table> 

   ### Upsampling 

We recomend to follow the file <b> ./segmentation_and_rebuild/main.sh </b> as guide using the sample dataset <b><i>forest4D_dlr</t></b>.

The code <b>./partition/write_segments_to_pc.py </b> has the following arguments:
    
#### Required arguments.

 <table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--path_to_data</td>
    <td>"../Data/forest4D_dlr"</td>
    <td>str</td>
    <td>Path to data</td>
  </tr>
  <tr>
    <td>--path_to_output</td>
    <td>"../Data/forest4D_dlr"</td>
    <td>str</td>
    <td>Directory to store results</td>
  </tr>
  <tr>
    <td>--areas</td>
    <td>"testing/, training/, validation/"</td>
    <td>str</td>
    <td>areas to be processed</td>
  </tr>
      <tr>
    <td>--n_classes</td>
    <td>4</td>
    <td>int</td>
    <td>number of classes</td>
  </tr>
      <tr>
    <td>--metrics</td>
    <td>False</td>
    <td>bool</td>
    <td>Compute metrics</td>
  </tr>
  <tr>
    <td>--file_extension</td>
    <td>".txt"</td>
    <td>str</td>
    <td>file extension default txt</td>
  </tr>
  <tr>
    <td>--gt_index</td>
    <td>3</td>
    <td>int</td>
    <td>ground true index in file</td>
  </tr>
      <tr>
    <td>--ver_batch</td>
    <td>500000</td>
    <td>int</td>
    <td>Batch size for reading large files</td>
  </tr>
</table>   


### Training and Inference

We recomend to follow the file <b> ./semantic_segmentation/main.sh </b> as guide using the sample dataset <b><i>forest4D_dlr</t></b>.

The code <b>./semantic_segmentation/gpu_main.py </b> has the following arguments:


- ### Computation configuration
#### Required arguments 

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--training</td>
    <td>True</td>
    <td>bool</td>
    <td>True:Training, False:Testing </td>
  </tr>
  <tr>
    <td>--only_test</td>
    <td>True</td>
    <td>bool</td>
    <td>True:Evaluate test, False:evaluate training and testing </td>
  </tr> 
  <tr>
    <td>--batch_size</td>
    <td>1024</td>
    <td>int</td>
    <td>Batch size for training </td>
  </tr>  
  <tr>
    <td>--n_epochs</td>
    <td>1000</td>
    <td>int</td>
    <td>Number of epochs </td>
  </tr>
  <tr>
    <td>--freq_aug</td>
    <td>5</td>
    <td>int</td>
    <td>Frequency in epochs of training augmentation </td>
  </tr>
  <tr>
    <td>--freq_validation</td>
    <td>5</td>
    <td>int</td>
    <td>Frequency validation in epochs </td>
  </tr>
  <tr>
    <td>--num_gpus</td>
    <td>1</td>
    <td>int</td>
    <td>How many GPUs to use </td>
  </tr>  
  <tr>
    <td>--gpu_memory_fraction</td>
    <td>.5</td>
    <td>int</td>
    <td>Fraction of the GPU to be allocated to that process, -1 to choose automatically </td>
  </tr>
</table> 

- ### Dataset    
#### Required arguments 

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--odir</td>
    <td>"log"</td>
    <td>str</td>
    <td>Output directory to store logs</td>
  </tr>
  <tr>
    <td>--log_file</td>
    <td>"log_doc"</td>
    <td>str</td>
    <td>Log file to store results</td>
  </tr>
  <tr>
    <td>--db_train_name</td>
    <td>"training"</td>
    <td>str</td>
    <td>Training folder</td>
  </tr>
  <tr>
    <td>--db_test_name</td>
    <td>"testing"</td>
    <td>str</td>
    <td>Testing or validation folder</td>
  </tr>
    <tr>
    <td>--SEMA3D_PATH</td>
    <td>'../Data/forest4D_dlr'</td>
    <td>str</td>
    <td>Dataset directory</td>
  </tr>
    <tr>
    <td>--model_name</td>
    <td>'dlr_new_model'</td>
    <td>str</td>
    <td>model_name</td>
  </tr>
    
</table>   

   
#### Optional arguments.

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--dataset</td>
    <td>'other'</td>
    <td>str</td>
    <td>'Dataset name: sema3d|s3dis'</td>
  </tr>
  <tr>
    <td>--resume</td>
    <td>True</td>
    <td>bool</td>
    <td>True: load a previously saved model</td>
  </tr>
  <tr>
    <td>--resume_best_val</td>
    <td>False</td>
    <td>bool</td>
    <td>True: load model from best validation result</td>
  </tr>
  <tr>
    <td>--restoring_partially</td>
    <td>False</td>
    <td>bool</td>
    <td>True: to initiallize with pretraining model</td>
  </tr>
   <tr>
    <td>--pre_train</td>
    <td>'dlr_old_model'</td>
    <td>str</td>
    <td>"pretrain model to initialize new model"</td>
  </tr>
</table>   


- ### Point cloud pre-processing
#### Optional arguments 

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--pc_xyznormalize</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]</td>
  </tr>
   <tr>
    <td>--pc_augm</td>
    <td>False</td>
    <td>bool</td>
    <td>Training augmentation</td>
  </tr>
  <tr>
    <td>--pc_augm_scale</td>
    <td>1.1</td>
    <td>float</td>
    <td>Training augmentation: Uniformly random scaling in [1/scale, scale] </td>
  </tr>
   <tr>
    <td>--pc_augm_rot</td>
    <td>True</td>
    <td>bool</td>
    <td>Training augmentation: Bool, random rotation around z-axis </td>
  </tr>
   <tr>
    <td>--pc_augm_mirror_prob</td>
    <td>False</td>
    <td>bool</td>
    <td>Training augmentation: Probability of mirroring about x or y axes</td>
  </tr>
     <tr>
    <td>--pc_augm_jitter</td>
    <td>False</td>
    <td>bool</td>
    <td>Training augmentation: Bool, Gaussian jittering of all attributes </td>
  </tr>
     <tr>
    <td>--spg_attribs01</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool, normalize edge features to 0 mean 1 deviation </td>
  </tr>    
</table>   

- ### Model configuration
#### Required arguments 

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--n_classes</td>
    <td>3</td>
    <td>int</td>
    <td>Number of classes</td>
  </tr>
  <tr>
    <td>--pc_xyznormalize</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool</td>
  </tr>
  <tr>
    <td>--pc_xyznormalize</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool</td>
  </tr>
  <tr>
    <td>--pc_xyznormalize</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool</td>
  </tr>
  <tr>
    <td>--pc_xyznormalize</td>
    <td>True</td>
    <td>bool</td>
    <td>Bool</td>
  </tr>



</table>   

#### Optional arguments 

<table style="width:100%">
  <tr>
    <th>Argument</th>
    <th>default</th>
    <th>type</th> 
    <th>help</th>
  </tr>
  <tr>
    <td>--ptn_minpts</td>
    <td>40</td>
    <td>int</td>
    <td>Minimum number of points into a segment for computing its embedding</td>
  </tr>
  <tr>
    <td>--ptn_npts</td>
    <td>512</td>
    <td>bool</td>
    <td>Maximum number of points into a segment for computing its embedding</td>
  </tr>
    <tr>
    <td>--rgb</td>
    <td>False</td>
    <td>bool</td>
    <td>Consider RGB on Training</td>
  </tr>
      <tr>
    <td>--learning_rate</td>
    <td>0.01</td>
    <td>float</td>
    <td>initial learning rate for training</td>
  </tr>
 <tr>
    <td>--class_weights</td>
    <td>False</td>
    <td>bool</td>
    <td>Compute class weights for imbalanced datasets</td>
  </tr>
     <tr>
    <td>--sor</td>
    <td>False</td>
    <td>bool</td>
    <td>Statistical Outlier Removal</td>
  </tr>
  <tr>
    <td>--mean_k</td>
    <td>30</td>
    <td>float</td>
    <td>mean_k for Statistical Outlier Removal</td>
  </tr>
  <tr>
    <td>--std_dev</td>
    <td>1.0</td>
    <td>float</td>
    <td>std_dev for Statistical Outlier Removal</td>
  </tr>
  <tr>
  <td>--l2_norm</td>
    <td>False</td>
    <td>bool</td>
    <td>l2 normalization</td>
  </tr>    
</table> 
