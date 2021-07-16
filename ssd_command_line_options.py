#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from convolutional_base_options import create_resnet
import object_detection_config


# In[2]:


def lr_scheduler(epoch):
    
    """Function to implement learning rate scheduler to be used in Keras's builtin or Custom Callbacks"""
    
    lr = 1e-3
    epoch_offset = object_detection_config.params['epoch_offset']
    if epoch > (200 - epoch_offset):
        lr *= 1e-4
    elif epoch > (180 - epoch_offset):
        lr *= 5e-4
    elif epoch > (160 - epoch_offset):
        lr *= 1e-3
    elif epoch > (140 - epoch_offset):
        lr *= 5e-3
    elif epoch > (120 - epoch_offset):
        lr *= 1e-2
    elif epoch > (100 - epoch_offset):
        lr *= 5e-2
    elif epoch > (80 - epoch_offset):
        lr *= 1e-1
    elif epoch > (60 - epoch_offset):
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr


# In[3]:


def our_ssd_cnn_parser():

    """Instatiate a command line parser for building, training, and testing of ssd network model"""
    
    parser = argparse.ArgumentParser(description='Object detection using SSD')
    # arguments for model building and training
    desc = "Number of extra layers in the SSD CNN after the prebuilt or custom resnet first stage convolutional base"
    parser.add_argument("--num_extra_layers",default=2,type=int,help=desc)
    
    desc = "Batch size to be used during training of SSD CNN"
    parser.add_argument("--batch_size",default=2,type=int,help=desc)
    
    desc = "Number of epochs to train SSD CNN"
    parser.add_argument("--epochs",default=50,type=int,help=desc)
    
    desc = "Number of data generator worker threads"
    parser.add_argument("--workers",default=3,type=int,help=desc)
    
    desc = "IoU threshold to be used in second round for fetching more positive anchor boxes"
    parser.add_argument("--iou_threshold",default=0.6,type=float,help=desc)
    
    desc = "Convolutional base of the first stage to be used in the SSD CNN"
    parser.add_argument("--conv_base",default=create_resnet,help=desc)
    
    desc = "Will train the model"
    parser.add_argument("--train",action='store_true',help=desc)
    
    desc = "Print model summary (text and png)"
    parser.add_argument("--summary",default=False,action='store_true',help=desc)
    
    desc = "Use categorical focal and smooth L1 loss functions"
    parser.add_argument("--improved_loss",default=False,action='store_true',help=desc)
    
    desc = "Use smooth L1 loss function"
    parser.add_argument("--smooth_l1",default=False,action='store_true',help=desc)
    
    desc = "Use normalized predictions"
    parser.add_argument("--is_normalize",default=False,action='store_true',help=desc)
    
    desc = "Directory for saving files"
    parser.add_argument("--save_dir",default="weights",help=desc)
    
    desc = "Dataset used to train the SSD CNN"
    parser.add_argument("--dataset",default="ExDark",help=desc)

    # inputs configurations
    desc = "Input image height"
    parser.add_argument("--height",default=500,type=int,help=desc)
    
    desc = "Input image width"
    parser.add_argument("--width",default=375,type=int,help=desc)
    
    desc = "Input image channels"
    parser.add_argument("--channels",default=3,type=int,help=desc)

    # dataset configurations
    desc = "Path to dataset directory"
    parser.add_argument("--data_path",default="dataset/ExDark",help=desc)
    
    desc = "Train labels csv file name"
    parser.add_argument("--train_labels",default="training_data_gt_labels.csv",help=desc)
    
    desc = "Test labels csv file name"
    parser.add_argument("--test_labels",default="testing_data_gt_labels.csv",help=desc)

    # configurations for evaluation of a trained model
    desc = "Load h5 model trained weights"
    parser.add_argument("--restore_weights",help=desc)
    
    desc = "Evaluate model"
    parser.add_argument("--evaluate",default=False,action='store_true',help=desc)
    
    desc = "Image for evaluation"
    parser.add_argument("--image_file",default=None,help=desc)
    
    desc = "Class posterior probability threshold while applying NMS over the predictions of a network"
    parser.add_argument("--posterior_prob_threshold",default=0.5,type=float,help=desc)
    
    desc = "IoU threshold while performing NMS"
    parser.add_argument("--nms_iou_threshold",default=0.2,type=float,help=desc)
    
    return parser

