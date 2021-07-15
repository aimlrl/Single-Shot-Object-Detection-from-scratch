#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import math
from skimage.io import imread
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from anchor_boxes_funcs import generate_anchor_boxes, corner_to_center, center_to_corner, compute_iou
from training_data_funcs import determine_anchor_box_color, index2class


# In[19]:


def non_maximum_suppression(user_args,predicted_cls,predicted_off,anchor_boxes_info):
    
    """This function performs Non Maximum Suppression on the predictions of SSD. 
    
    Parameters: 
                user_args: User defined arguments
                predicted_cls (tensor) : Softmax probabilities over different categories for different 
                                         anchor boxes
                predicted_off (tensor) : Predicted bounding box offsets for different anchor boxes
                anchor_boxes_info (tensor) : Complete information about all the anchor boxes per scale
                
    Returning:
                anchor_predicted_cls (tensor) : predicted class per anchor box
                non_bg_anchor_box_indices (tensor) : non background class anchor box indices filtered by 
                                                     NMS
                posterior_prob (tensor) : posterior probabilities non background class indexes filtered by 
                                          NMS
    """
    #Converting all softmax probabilities over categories for different anchor boxes into predicted class 
    #indexes
    anchor_predicted_cls = np.argmax(predicted_cls,axis=1)
    
    #Seperating all the predicted non background class anchor boxes from different anchor boxes
    non_background_cls = np.nonzero(anchor_predicted_cls)[0]
    
    #Going to create a list which will hold the anchor boxes having non background categories
    non_bg_anchor_box_indices = []
    
    while True:
        
        #Numpy array of zero softmax posterior probabilities for every anchor box
        posterior_prob = np.zeros((predicted_cls.shape[0],))
        
        #Filling up highest posterior probability values for all anchors having non background classes
        posterior_prob[non_background_cls] = np.amax(predicted_cls[non_background_cls],axis=1)
        
        #Fetching the anchor boxes having highest probability over different assigned categories
        max_posterior_prob_idx = np.argmax(posterior_prob,axis=0)
        max_posterior_prob = posterior_prob[max_posterior_prob_idx]
        
        #Now, the remaining anchor boxes having non background classes are fetched
        non_background_cls = non_background_cls[non_background_cls != max_posterior_prob_idx]
        
        #Let's see that whether our filtered highest poseterior probability non background anchor boxes will
        #be able to survive the high confidence threshold set by user somewhere between (0.8,0.9).
        
        if max_posterior_prob < user_args.posterior_prob_threshold:
        #if max_posterior_prob < args.posterior_prob_threshold:
            break
            
        non_bg_anchor_box_indices.append(max_posterior_prob_idx)
        filtered_anchor_boxes = anchor_boxes_info[max_posterior_prob_idx]
        filtered_off = predicted_off[max_posterior_prob_idx][0:4]
        
        filtered_anchor_box_desc = filtered_anchor_boxes + filtered_off
        filtered_anchor_box_desc = np.expand_dims(filtered_anchor_box_desc,axis=0)
        
        non_background_cls_copy = np.copy(non_background_cls)
        
        #Going to go through the step of computing IoU of the remaining anchor boxes which were not able to
        #survive high confidence threshold with the ones which were able to survive, for every class. 
        
        for non_background_idx in non_background_cls_copy:
            
            anchor_box = anchor_boxes_info[non_background_idx]
            offset = predicted_off[non_background_idx]
            
            anchor_box_desc = anchor_box + offset
            anchor_box_desc = np.expand_dims(anchor_box_desc,axis=0)
            
            iou = compute_iou(anchor_box_desc,filtered_anchor_box_desc)[0][0]
            
            if iou >= user_args.nms_iou_threshold:
            #if iou >= args.iou_threshold:
                non_background_cls = non_background_cls[non_background_cls != non_background_idx]
                
        if non_background_cls.size == 0:
            break
            
    posterior_prob = np.zeros((predicted_cls.shape[0],))
    posterior_prob[non_bg_anchor_box_indices] = np.amax(predicted_cls[non_bg_anchor_box_indices],axis=1)
    return anchor_predicted_cls,non_bg_anchor_box_indices,posterior_prob


# In[20]:


def display_predicted_boxes(args,frame,predicted_cls,predicted_off,feature_maps_shapes,isdisplay=True):
    """This function displays the pixels enclosed by several bounding boxes in an image.
    
    Parameters:
                args: User Defined arguments
                frame (tensor): image having detected objects
                predicted_cls (tensor): Softmax probabilities over categories for different anchor boxes
                predicted_off (tensor): Predicted bounding box offsets for different anchor boxes
                feature_maps_shapes (tensor): shapes of feature maps from multi scale stages of SSD
                isdisplay (bool): Should predicted bounding boxes be shown or not
                
    Returns:
                category_strings (list): List of different class strings
                anchor_coordinates (list): List of bounding boxes to be drawn in the form of Matplotlib rectangles
                predicted_cls_int_labels (list): Integer labels of predicted categories
                filtered_anchor_boxes (list): filtered anchor boxes of detected objects
    """
    #Let's generate all anchor boxes for all feature maps 
    #(Basically, we will get the details of all anchor boxes in the Image)
    multiscale_levels = len(feature_maps_shapes)
    
    for multiscale_idx, map_shape in enumerate(feature_maps_shapes):
        total_anchor_boxes_per_fmap = generate_anchor_boxes(map_shape,frame.shape,
                                                           multiscale_index=multiscale_idx)
        total_anchor_boxes_per_fmap = total_anchor_boxes_per_fmap.reshape((-1,4))
        
        if multiscale_idx == 0:
            total_anchor_boxes_all_fmaps = total_anchor_boxes_per_fmap
        else:
            total_anchor_boxes_all_fmaps = np.concatenate((total_anchor_boxes_all_fmaps,
                                                           total_anchor_boxes_per_fmap),axis=0)
            
        print(map_shape)
        
    if args.is_normalize == True:
        norm_total_anchor_boxes_all_fmaps = corner_to_center(total_anchor_boxes_all_fmaps)
        #The standard deviation in offsets of centroid is usually 10%
        predicted_off[:,0:2] = 0.1*predicted_off[:,0:2]
        predicted_off[:,0:2] = predicted_off[:,0:2] * norm_total_anchor_boxes_all_fmaps[:,2:4]
        predicted_off[:,0:2] = predicted_off[:,0:2] + norm_total_anchor_boxes_all_fmaps[:,0:2]
        
        #The standard deviation in offsets of width and height is usually 20%
        predicted_off[:,2:4] = 0.2*predicted_off[:,2:4]
        predicted_off[:,2:4] = np.exp(predicted_off[:,2:4])
        predicted_off[:,2:4] = predicted_off[:,2:4] * norm_total_anchor_boxes_all_fmaps[:,2:4]
        
        #Now, we need to convert the cx,cy,w,h back to x,y,w,h because we need to perform NMS
        predicted_off = center_to_corner(predicted_off)
        predicted_off[:,0:4] = predicted_off[:,0:4] - total_anchor_boxes_all_fmaps
        
    anchor_predicted_cls, filtered_class_indices, posterior_prob = non_maximum_suppression(args,predicted_cls,
                                                                    predicted_off,
                                                                    total_anchor_boxes_all_fmaps)
    category_strings = []
    anchor_coordinates = []
    predicted_cls_int_labels = []
    filtered_anchor_boxes = []
    
    if isdisplay == True:
        #Writing the code to draw rectangular bounding boxes
        fig, axes = plt.subplots(1)
        axes.imshow(frame)
    h_offset = 1
    
    for idx in filtered_class_indices:
        filtered_anchor_box = total_anchor_boxes_all_fmaps[idx]
        filtered_offset = predicted_off[idx]
        filtered_anchor_box = filtered_anchor_box + filtered_offset[0:4]
        filtered_anchor_boxes.append(filtered_anchor_box)
        w = filtered_anchor_box[1] - filtered_anchor_box[0]
        h = filtered_anchor_box[3] - filtered_anchor_box[2]
        x = filtered_anchor_box[0]
        y = filtered_anchor_box[2]
        
        predicted_cls_int_labels.append(int(anchor_predicted_cls[idx]))
        category_name = index2class(int(anchor_predicted_cls[idx]))
        category_name = "%s: %0.2f" % (category_name,posterior_prob[idx])
        category_strings.append(category_name)
        anchor_box_coordinates = (x,y,w,h)
        
        print(category_name,anchor_box_coordinates)
        
        anchor_coordinates.append(anchor_box_coordinates)
        
        if isdisplay == True:
            color = determine_anchor_box_color(int(anchor_predicted_cls[idx]))
            drawn_rectangle = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,facecolor='none')
            axes.add_patch(drawn_rectangle)
            
            anchor_box_text_settings = dict(color='white',alpha=1.0)
            
            axes.text(filtered_anchor_box[0] + 2, filtered_anchor_box[2]-16+np.random.randint(0,h_offset),
                     category_name,bbox=anchor_box_text_settings,fontsize=10,verticalalignment='top')
            
            h_offset = h_offset + 50
            
    if isdisplay == True:
        plt.savefig("Detection Result.png",dpi=600)
        plt.show()
        
    return category_strings,anchor_coordinates,predicted_cls_int_labels,filtered_anchor_boxes


# In[21]:


def debug_predicted_boxes(frame,feature_maps,anchor_boxes,maximum_iou_abs=None,maximum_iou_per_cat=None,
                         gt_bboxes=None,is_abs_display=True):
    """A function for testing the working of Non Maximum Suppression.
    
    Parameters:
                frame (tensor) : Image in which to detect objects
                feature_maps (list or a tuple) : Shape of the feature maps
                anchor_boxes (tensor) : description (desc) of all anchor boxes generated
                maximum_iou_abs (tensor) : all the anchor boxes having maximum iou
                maximum_iou_par_cat (tensor) : all the anchor boxes per ground truth category having 
                                               maximum iou
                gt_bboxes (tensor) : Coordinates of Ground truth bounding boxes
                is_abs_display (bool) : Whether anchor boxes should be displayed or not
                
    Returning:
                fig (object) : figure object related to matplotlib
                axes (different axis objects) : different axes on which to plot the boxes
    """
    
    frame_height,frame_width,_ = frame.shape
    _,feature_map_height,feature_map_width,_ = feature_maps
    
    fig,axes = plt.subplots(1)
    axes.imshow(frame)
    
    if is_abs_display == True:
        anchor_box_height = frame_height//feature_map_height
        
        for i in range(feature_map_height):
            height = i*anchor_box_height
            horizontal_line = Line2D([0,frame_width],[height,height])
            axes.add_line(horizontal_line)
            
        anchor_box_width = frame_width//feature_map_width
        
        for i in range(feature_map_width):
            width = i*anchor_box_width
            vertical_line = Line2D([width,width],[0,frame_width])
            axes.add_line(vertical_line)
        
    for idx in range(maximum_iou_abs.shape[1]):
        row = maximum_iou_abs[1,idx]
        col = maximum_iou_abs[2,idx]
        ab = maximum_iou_abs[3,idx]
        
        anchor_box_coordinates = anchor_boxes[0,row,col,ab]
        
        w = anchor_box_coordinates[1] - anchor_box_coordinates[0]
        h = anchor_box_coordinates[3] - anchor_box_coordinates[2]
        x = anchor_box_coordinates[0]
        y = anchor_box_coordinates[2]
        
        drawn_rectangle = Rectangle((x,y),w,h,linewidth=2,edgecolor='y',facecolor='none')
        
        axes.add_patch(drawn_rectangle)
        
        if maximum_iou_per_cat is not None and gt_bboxes is not None:
            
            iou = np.amax(maximum_iou_per_cat[idx])
            gt_bbox = gt_bboxes[idx]
            
            category_name = index2class(int(gt_bbox[4]))
            color = determine_anchor_box_color(int(gt_bbox[4]))
            anchor_box_text_settings = dict(facecolor=color,color=color,alpha=1.0)
            
            axes.text(gt_bbox[0], gt_bbox[2],
                     category_name,bbox=anchor_box_text_settings,fontsize=16,verticalalignment='top',
                      color='w',fontweight='bold')
            
            delta_xmin = gt_bbox[0] - anchor_box_coordinates[0]
            delta_xmax = gt_bbox[1] - anchor_box_coordinates[1]
            delta_ymin = gt_bbox[2] - anchor_box_coordinates[2]
            delta_ymax = gt_bbox[3] - anchor_box_coordinates[3]
            
            print(idx,":","(",category_name,")",iou,delta_xmin,delta_xmax,delta_ymin,delta_ymax)
            
        if gt_bboxes is None:
            plt.show()
            
        return fig,axes

