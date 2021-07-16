#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber
import numpy as np


# In[43]:


def categorical_focal_loss(c,posterior_probabilities):

    """This function implements custom focal loss on standard categorical cross entropy loss"""

    gamma = 2.0
    alpha = 0.25

    categorical_cross_entropy = -c * K.log(posterior_probabilities)
    focal_loss_scaling = alpha * K.pow((1-posterior_probabilities),gamma)

    scaled_cross_entropy = focal_loss_scaling * categorical_cross_entropy

    focal_loss_cross_entropy = K.mean(K.sum(K.sum(scaled_cross_entropy,axis=-1),axis=1))

    return focal_loss_cross_entropy


# In[44]:


def positive_anchor_boxes_offset(y_gt,y_hat):
    
    """This function preprocesses offsets of ground truth bounding boxes as well as predicted offsets"""
    
    gt_offsets = y_gt[...,0:4]
    positive_anchor_boxes_indicator = y_gt[...,4:8]
    
    predicted_offsets = y_hat[...,0:4]
    
    gt_offsets = gt_offsets * positive_anchor_boxes_indicator
    predicted_offsets = predicted_offsets * positive_anchor_boxes_indicator
    
    return gt_offsets,predicted_offsets


# In[80]:


def l1_loss(y_gt,y_hat):
    
    """This function implements L1 Loss which is also called Mean Absolute Error Loss"""
    
    gt_offsets,predicted_offsets = positive_anchor_boxes_offset(y_gt,y_hat)
    
    return K.mean(K.sum(K.sum(K.abs(predicted_offsets - gt_offsets),axis=-1),axis=1))


# In[81]:


def smooth_l1_loss(y_gt,y_hat):
    
    """This function implements the smoothen out version of L1 loss so that it becomes differentiable"""
    
    gt_offsets,predicted_offsets = positive_anchor_boxes_offset(y_gt,y_hat)
    
    return Huber()(gt_offsets,predicted_offsets)

