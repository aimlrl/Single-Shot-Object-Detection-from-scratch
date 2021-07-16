#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randint
import object_detection_config
from matplotlib.patches import Rectangle


# In[2]:


def training_data_validation_conversion(file_path):
    
    """This function reads a csv file of Training Data. In the case of object detection, the training data
    will consist of training examples where each training example (row of a csv file) will consist of the
    following entities:
    
    First, Feature Vector (Image Pixels or File name of the image, in our case it's file name of the image)
    
    Second, Ground truth bounding box coordinates of the object present in the image
    
    Last, Class label of the correponding object present in the image
    
    Parameters:
                file_path (string) : File path of CSV file of Training Data
                
    Returning:
                training_data_dict (dict) : A dictionary of the training data where each key value pair
                                            in the dictionary will consist of key as File name of the image
                                            and the value will consist of list of bounding box coordinates
                                            of all ground truth bounding boxes as well as class labels of 
                                            all the objects present in the image.
    """
                                            
    training_data = list()
    
    with open(file_path) as file_handle:
        
        training_examples = csv.reader(file_handle)
        
        for example in training_examples:
            training_data.append(example)
            
    training_data = np.array(training_data)[1:]
    file_names = training_data[:,0]
    training_data_dict = dict()
    unique_file_names = np.unique(file_names)
    total_unique_classes = np.unique(training_data[:,-1]).astype(int).tolist()
    
    for image_file in unique_file_names:
        training_data_dict[image_file] = list()
        
    invalid_example = False
    
    for training_example in training_data:
        gt_info = training_example[1:]
        
        if len(training_example) != 6:
            invalid_example = True
            print("In image file name",training_example[0],"there is incomplete ground truth information")
            
        elif gt_info[0] == gt_info[1] or gt_info[2] == gt_info[3]:
            invalid_example = True
            print("In image file name",training_example[0],"Zero width or Height ground truth bounding box is found")
            
        elif gt_info[-1] == 0:
            invalid_example = True
            print("In image file name",training_example[0],"no background category is found")
            
        if invalid_example == False:
            gt_info = gt_info.astype(np.float32)
            training_data_dict[training_example[0]].append(gt_info)
            
        invalid_example = False
        
    for training_example in training_data:
        if len(training_data_dict[training_example[0]]) == 0:
            del training_data_dict[training_example[0]]
            
    return training_data_dict,total_unique_classes


# In[3]:


def determine_anchor_box_color(index=None):
    """This function determines the color of the bounding box for a specific category object.
    
    Parameters:
                index : Index of the object category
    Returning:
                color_character : Matplotlib compatible color character for a specific object category index
    """
    colors = ['w','r','b','g','c','m','y','g','c','m','k']
    
    if index is None:
        return colors[randint(0,len(colors)-1)]
    return colors[index % len(colors)]


# In[4]:


def show_colored_pred_bboxes(frame, pred_info, ax=None):
    """This function draws a bounding box of a specific color based on the predicted category of an object 
    given it's predicted box coords.
    """
    
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(frame)
        
    for pred_off_cls in pred_info:
        w = pred_off_cls[1] - pred_off_cls[0]
        h = pred_off_cls[3] - pred_off_cls[2]
        x = pred_off_cls[0]
        y = pred_off_cls[2]
        pred_object_index = int(pred_off_cls[4])
        color = determine_anchor_box_color(pred_object_index)
        rect_to_draw = Rectangle((x, y),w,h,linewidth=2,edgecolor=color,facecolor='none')
        ax.add_patch(rect_to_draw)
    plt.show()


# In[5]:


def index2class(index=0):
    """Convert index (int) to class name (string)"""
    classes = object_detection_config.params['classes']
    return classes[index]


# In[6]:


def class2index(category="Background"):
    """Convert class name (string) to index (int)"""
    classes = object_detection_config.params['classes']
    return classes.index(category)

