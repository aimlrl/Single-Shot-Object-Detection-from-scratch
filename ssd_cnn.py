#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.layers import Activation,Dense,Input
from keras.layers import Conv2D,Flatten
from keras.layers import BatchNormalization
from keras.layers import Reshape,Concatenate
from keras.models import Model
from keras import backend as K
import numpy as np
from convolutional_base_options import create_resnet
from keras.layers import MaxPooling2D


# In[3]:


def ith_scale_output_layer(num_filters,ksize,stride,padding):

    """This function creates an output layer for the different scales at which the detection needs to be performed.

    Parameters:
                  num_filters (int) : Number of filters to be used in a convolutional layer

    Returning:
                  a model layer
    """

    conv_layer = Conv2D(filters=num_filters,kernel_size=ksize,padding=padding)

    return conv_layer


# In[4]:


def build_ssd(input_shape,convolutional_base,num_extra_layers,num_classes,aspect_ratios=[1,2,0.5]):

    """This function builds the complete Neural Network architecture for Single Shot Detection (SSD) 
       on the top of convolutional base. 
       
    Parameters:
                  input_shape (tuple) : Shape of the input image to be fed into the Neural Network of SSD
                  convolutional_base : Keras model based on functional API
                  num_extra_layers (int) : Number of extra layers added in the convolutional base for 
                                           multiscale object detection
                  num_classes (int) : Number of classes in the training data
                  aspect_ratios (int) : Number of aspect ratios required for every anchor box

    Returning:
                  num_anchor_boxes_per_fmap_pt (int) : Number of anchor boxes per feature map point
                  ssd_cnn_model : Keras model of complete Neural Network architecture for 
                                  Single Shot Detection (SSD)
    """

    num_anchor_boxes_per_fmap_pt = len(aspect_ratios) + 1
    input_images = Input(shape=input_shape)
    conv_base_outputs = convolutional_base(input_images)
    num_outputs = num_extra_layers + 1

    ssd_cnn_cls_outputs = []
    ssd_cnn_offset_outputs = []
    prebuilt_output_layer_configs = [(1,1,"valid"),(3,1,"valid"),(1,1,"same")]
    custom_output_layer_configs = [(5,1,"valid"),(3,1,"valid"),(1,1,"same")]

    for i in range(num_outputs):
        ith_scale_output_fmap = conv_base_outputs[i]

        if convolutional_base.name == "Prebuilt":
            ksize,stride,padding = prebuilt_output_layer_configs[i]
        else:
            ksize,stride,padding = custom_output_layer_configs[i]

        if convolutional_base.name == "Prebuilt" and i == 0:
            ith_scale_class_pred = ith_scale_output_layer(num_filters=num_anchor_boxes_per_fmap_pt*num_classes,
                                                          ksize=ksize,stride=stride,padding=padding)(ith_scale_output_fmap)
            ith_scale_class_pred = MaxPooling2D()(ith_scale_class_pred)
            ith_scale_offset_pred = ith_scale_output_layer(num_filters=4*num_anchor_boxes_per_fmap_pt,
                                                           ksize=ksize,stride=stride,padding=padding)(ith_scale_output_fmap)
            ith_scale_offset_pred = MaxPooling2D()(ith_scale_offset_pred)
        else:
            ith_scale_class_pred = ith_scale_output_layer(num_filters=num_anchor_boxes_per_fmap_pt*num_classes,ksize=ksize,stride=stride,padding=padding)(ith_scale_output_fmap)
            ith_scale_offset_pred = ith_scale_output_layer(num_filters=4*num_anchor_boxes_per_fmap_pt,ksize=ksize,stride=stride,padding=padding)(ith_scale_output_fmap)
                                                                                                       
        ith_scale_class_pred = Reshape((-1,num_classes))(ith_scale_class_pred)
        ith_scale_offset_pred = Reshape((-1,4))(ith_scale_offset_pred)

        ssd_cnn_offset_outputs.append(ith_scale_offset_pred)
        ith_scale_class_pred = Activation("softmax")(ith_scale_class_pred)
        ssd_cnn_cls_outputs.append(ith_scale_class_pred)

    ssd_cnn_cls_outputs = Concatenate(axis=1)(ssd_cnn_cls_outputs)
    ssd_cnn_offset_outputs = Concatenate(axis=1)(ssd_cnn_offset_outputs)

    all_ssd_cnn_outputs = [ssd_cnn_cls_outputs,ssd_cnn_offset_outputs]

    ssd_cnn_model = Model(inputs=input_images,outputs=all_ssd_cnn_outputs)

    return num_anchor_boxes_per_fmap_pt,ssd_cnn_model


# In[ ]:




