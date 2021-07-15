#!/usr/bin/env python
# coding: utf-8

# In[29]:


from tensorflow.keras.utils import Sequence
import numpy as np
import os
from skimage import img_as_float
from matplotlib.pyplot import imread
from anchor_boxes_funcs import generate_anchor_boxes,compute_iou,fetch_all_positive_anchor_boxes


# In[30]:


class custom_dataset_data_generator(Sequence):
    
    """This class will instantiate the custom image data generator based on root class (Sequence) for all 
    the data generators in Keras
    
    Attributes:
                args : user defined configuration
                training_data_dictionary (dict): Dictionary having keys as image filenames and values as 
                                            Ground truth labels of images (Class labels and Bounding boxes)
                total_training_classes (int): Number of unique object categories in training data
                feature_map_shapes (list): List of shapes of feature maps given as an output from different 
                                           layers of a SSD CNN
                num_anchor_boxes_per_fmap_pt (int) : Number of aspect ratios of an anchor box including the
                                                     new scale for aspect ratio=1 (square root one)
                is_shuffle (bool) : Whether to shuffle the dataset or not
    """
    
    def __init__(self,args,training_data_dictionary,total_training_classes,feature_map_shapes=[],
                num_anchor_boxes_per_fmap_pt=4,is_shuffle=True):
        
        self.user_args = args
        self.training_data_dict = training_data_dictionary
        self.unique_classes = total_training_classes
        self.image_file_names = np.array(list(self.training_data_dict.keys()))
        self.input_shape = (self.user_args.width,self.user_args.height,self.user_args.channels)
        self.fmap_shapes = feature_map_shapes
        self.num_aspect_ratios = num_anchor_boxes_per_fmap_pt
        self.shuffle = is_shuffle
        self.on_epoch_end()
        
    def __len__(self):
        
        """This function will calculate the total number of mini batches per epoch"""
        
        num_batches_per_epoch = np.floor(len(self.training_data_dict)/self.user_args.batch_size)
        return int(num_batches_per_epoch)
    
    def __getitem__(self,index):
        
        """This function will generate one mini batch worth of data."""
        
        start_index = index * self.user_args.batch_size
        end_index = (index + 1) * self.user_args.batch_size
        img_file_names = self.image_file_names[start_index : end_index]
        
        img_pixels,gt_labels = self.__data_generation(img_file_names)
        
        return img_pixels,gt_labels
    
    
    def __data_generation(self,img_file_names):
        
        """This function is responsible for the generation of training data in proper format for 
            training our SSD CNN by generating images from their file names as well as their ground
            truth labels.
            
        Parameters:
                    img_file_names (tensor) : Randomly sampled names of the image files having total
                                              number of elements equal to the number of elements in mini
                                              batch. 
        
        Returning:
                    img_pixels (tensor) : Mini batch of image pixels
                    gt_labels (tensor) : Mini batch of Ground truth labels (Class labels of all the objects
                                         present in the images as well as ground truth bounding box 
                                         coordinates of all the objects present in the images.
        """
        
        img_pixels = np.zeros((self.user_args.batch_size,*self.input_shape))
        
        self.total_anchor_boxes = 0
        
        for fmap_shape in self.fmap_shapes:
            fmap_shape = list(fmap_shape)
            fmap_shape.append(self.num_aspect_ratios)
            self.total_anchor_boxes = self.total_anchor_boxes + np.prod(np.array(fmap_shape))
            
        anchor_boxes_categories = np.zeros((self.user_args.batch_size,self.total_anchor_boxes,
                                            self.unique_classes))
        
        anchor_boxes_bbox_coords = np.zeros((self.user_args.batch_size,self.total_anchor_boxes,4))
        
        positive_anchor_boxes_indicator = np.zeros((self.user_args.batch_size,self.total_anchor_boxes,4))
        
        for i,im_file_name in enumerate(img_file_names):
            
            mini_batch_img_path = os.path.join(self.user_args.data_path,im_file_name)
            im_pixels = img_as_float(imread(mini_batch_img_path))
            
            img_pixels[i] = im_pixels
            
            img_gt_labels = self.training_data_dict[im_file_name]
            img_gt_labels = np.array(img_gt_labels)
            
            gt_bboxes_coords = img_gt_labels[:,0:-1]
            
            for multiscale_idx, fmap_shape in enumerate(self.fmap_shapes):
                
                anchor_boxes_per_scale = generate_anchor_boxes(feature_map_shape=fmap_shape,
                                                              frame_shape=self.input_shape,
                                                              multiscale_index=multiscale_idx,
                                                              num_extra_layers=self.user_args.num_extra_layers)
                anchor_boxes_per_scale = np.reshape(anchor_boxes_per_scale,[-1,4])
                
                iou = compute_iou(anchor_boxes_per_scale,gt_bboxes_coords)
                
                pos_abs_cls,pos_abs_off,pos_abs_ind = fetch_all_positive_anchor_boxes(iou=iou,
                                                                          num_unique_categories=self.unique_classes,
                                                                         anchor_boxes_coords=anchor_boxes_per_scale,
                                                                         gt_info=img_gt_labels,
                                                                         is_normalize=self.user_args.is_normalize,
                                                                         iou_threshold=self.user_args.iou_threshold)
                
                if multiscale_idx == 0:
                    pos_cls = np.array(pos_abs_cls)
                    pos_off = np.array(pos_abs_off)
                    pos_ind = np.array(pos_abs_ind)
                else:
                    pos_cls = np.append(arr=pos_cls,values=pos_abs_cls,axis=0)
                    pos_off = np.append(arr=pos_off,values=pos_abs_off,axis=0)
                    pos_ind = np.append(arr=pos_ind,values=pos_abs_ind,axis=0)
                    
            anchor_boxes_categories[i] = pos_cls
            anchor_boxes_bbox_coords[i] = pos_off
            positive_anchor_boxes_indicator[i] = pos_ind
            
        gt_labels = [anchor_boxes_categories,np.concatenate([anchor_boxes_bbox_coords,
                                                            positive_anchor_boxes_indicator],axis=-1)]
        
        return img_pixels,gt_labels
    
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.image_file_names)

