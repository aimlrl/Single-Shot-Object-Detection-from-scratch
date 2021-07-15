#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


def generate_anchor_boxes_scales(num_extra_layers=2):
    """This function generates different scales of anchor boxes for extra convolutional layers 
    added to the pretrained convolutional base of any pretrained network. We are assuming that we have 
    added one more layer on the top of pretrained convolutional base, therefore it will result in object 
    detection on different objects on two different scales. 
    Scales : Sizes of anchor boxes in terms of Height and Width
    
    Parameters:
                num_extra_layers (int) : Number of convolutional layers added to the top of pretrained 
                                        convolutional base
    Returning:
                different_scales (list) : Different scales of the anchor boxes
    """
    
    scaling_factors = np.linspace(0.2,0.9,(num_extra_layers+2))
    #The three different scales considered by our detection are given by:
    #0.2 (the size of one anchor box will be 1/5th of the overall size of the image in both width and height)
    #0.55 (the size of one anchor box will be almost 1/2 of the overall size of the image in both width and height)
    #0.9 (the size of one anchor box will be 90 % of the overall size of the image in both width and height)
    
    different_scales = []
    
    for i in range(len(scaling_factors)-1):
        scale = [scaling_factors[i],math.sqrt(scaling_factors[i]*scaling_factors[i+1])]
        different_scales.append(scale)
        
    return different_scales


# In[3]:


def generate_anchor_boxes(feature_map_shape,frame_shape,multiscale_index=0,num_extra_layers=2,
                         aspect_ratios=[1,2,0.5]):
    """This function generates coordinates of anchor boxes on different aspect ratios for a given scale, 
    defined by the size of the feature map output by a specific convolutional layer (base or extra layer) 
    of the network.
    
    Parameters:
                feature_map_shape(list or a tuple) : Shape of the feature map as a result of output from
                                                     a convolutional layer
                frame_shape (list or tuple) : Shape of the frame which is given as an input to the network
                multiscale_index (int) : For which layer or scale of the convolutional layer, anchor boxes
                                         need to be fetched
                num_extra_layers (int) : Number of convolutional layers on the top of pretrained network
                aspect_ratios (list) : Number of aspect ratios which need to be taken into consideration
                                       for a specific scale of anchor boxes
    Returning:
                anchor_boxes_coords (tensor) : anchor boxes per feature map size as a result of output
                                                  from the convolutional layer of the network
    """
    #In order to generate anchor boxes for a specific convolutional layer (base layer or extra layer), 
    #first it's determined that what will be the scale of anchor boxes for that specific convolutional 
    #layer and then apply different aspect ratios on that scale.
    
    different_scales = generate_anchor_boxes_scales(num_extra_layers)
    anchor_box_scale_per_layer = different_scales[multiscale_index]
    
    aspect_ratios_per_anchor_box = len(aspect_ratios) + 1
    
    frame_height,frame_width,_ = frame_shape
    
    feature_map_height,feature_map_width = feature_map_shape
    
    #Now, we are going to finally get the scaled dimensions of the input image according to the specific 
    #convolutional layer (or scale)
    
    scaled_height = frame_height * anchor_box_scale_per_layer[0]
    scaled_width = frame_width * anchor_box_scale_per_layer[0]
    
    #Let's try to fetch different aspect ratio dimensions for the same scaled dimensions according to the 
    #specific convolutional layer
    
    all_aspect_ratios_per_scale = []
    
    for aspect_ratio in aspect_ratios:
        aspect_ratiod_width = scaled_width * math.sqrt(aspect_ratio)
        aspect_ratiod_height = scaled_height / math.sqrt(aspect_ratio)
        all_aspect_ratios_per_scale.append([aspect_ratiod_width,aspect_ratiod_height])
        
    #Finally, we have added all the aspect ratios according to the specific scale except for the 
    #alternative aspect ratio for 1. So, let's add alternative aspect ratio for 1. 
    
    aspect_ratiod_width = scaled_width * anchor_box_scale_per_layer[1]
    aspect_ratiod_height = scaled_height * anchor_box_scale_per_layer[1]
    all_aspect_ratios_per_scale.append([aspect_ratiod_width,aspect_ratiod_height])
    
    all_aspect_ratios_per_scale = np.array(all_aspect_ratios_per_scale)
    
    anchor_box_scale_width = frame_width / feature_map_width
    anchor_box_scale_height = frame_height / feature_map_height
    
    #Let's see how anchor box coordinates will be stored in the form of a tensor
    
    #Let's first find the x coordinate position of the top most left feature map point
    topmost_left_x = anchor_box_scale_width * 0.5
    
    #Let's now determine the x coordinate poistion of the top most right feature map point
    topmost_right_x = (feature_map_width * anchor_box_scale_width) - (0.5 * anchor_box_scale_width)
    
    #let's create x coordinate positions of feature map points between top most left feature map point 
    #and top most right feature map point at equally spaced intervals of anchor box width
    
    cx = np.linspace(topmost_left_x,topmost_right_x,feature_map_width)
    
    #Let's first find the y coordinate position of the top most left feature map point
    topmost_left_y = anchor_box_scale_height * 0.5
    
    #Let's now determine the y coordinate poistion of the top most right feature map point
    topmost_right_y = (feature_map_height * anchor_box_scale_height) - (0.5 * anchor_box_scale_height)
    
    #let's create y coordinate positions of feature map points between top most left feature map point 
    #and top most right feature map point at equally spaced intervals of anchor box height
    
    cy = np.linspace(topmost_left_y,topmost_right_y,feature_map_height)
    
    cx_grid, cy_grid = np.meshgrid(cx,cy)
    
    cx_grid = np.expand_dims(cx_grid,axis=-1)
    cy_grid = np.expand_dims(cy_grid,axis=-1)
    
    anchor_boxes_coords = np.zeros((feature_map_width,feature_map_height,
                                       aspect_ratios_per_anchor_box,4))
    
    anchor_boxes_coords[:,:,:,0] = np.tile(cx_grid,reps=(1,1,aspect_ratios_per_anchor_box))
    anchor_boxes_coords[:,:,:,1] = np.tile(cy_grid,reps=(1,1,aspect_ratios_per_anchor_box))
    anchor_boxes_coords[:,:,:,2] = all_aspect_ratios_per_scale[:,0]
    anchor_boxes_coords[:,:,:,3] = all_aspect_ratios_per_scale[:,1]
    
    return anchor_boxes_coords


# In[4]:


def corner_to_center(corner_coordinates):
    
    """This function will convert corner format into center format.
    That is from (xmin,xmax,ymin,ymax) to (cx,cy,w,h)
    
    Parameters:
                corner_coordinates (tensor) : Coordinates of boxes in corner format
    
    Returning:
                center_coordinates (tensor) : Coordinates of boxes in center format
    """
    center_coordinates = np.copy(corner_coordinates).astype(np.float)
    
    center_coordinates[...,0] = 0.5 *(corner_coordinates[...,1] - corner_coordinates[...,0])
    center_coordinates[...,0] = center_coordinates[...,0] + corner_coordinates[...,0]
    center_coordinates[...,1] = 0.5 *(corner_coordinates[...,3] - corner_coordinates[...,2])
    center_coordinates[...,1] = center_coordinates[...,1] + corner_coordinates[...,2]
    
    center_coordinates[...,2] = corner_coordinates[...,1] - corner_coordinates[...,0]
    center_coordinates[...,3] = corner_coordinates[...,3] - corner_coordinates[...,2]
    
    return center_coordinates    


# In[5]:


def center_to_corner(center_coordinates):
    
    """This function will convert center format into corner format.
    That is from (cx,cy,w,h) to (xmin,xmax,ymin,ymax)
    
    Parameters:
                center_coordinates (tensor) : Coordinates of boxes in center format
    
    Returning:
                corner_coordinates (tensor) : Coordinates of boxes in corner format
    """
    
    corner_coordinates = np.copy(center_coordinates).astype(np.float)
    
    corner_coordinates[...,0] = center_coordinates[...,0] - (0.5 * center_coordinates[...,2])
    corner_coordinates[...,1] = center_coordinates[...,0] + (0.5 * center_coordinates[...,2])
    corner_coordinates[...,2] = center_coordinates[...,1] - (0.5 * center_coordinates[...,3])
    corner_coordinates[...,3] = center_coordinates[...,1] + (0.5 * center_coordinates[...,3])
    
    return corner_coordinates


# In[6]:


def fetch_all_positive_anchor_boxes(iou,num_unique_categories,anchor_boxes_coords,gt_info,is_normalize=False,
                                   iou_threshold=0.6):
    """A function to fetch all the positive anchor boxes which are having maximum amount of overlap (IoU)  
    as well as IoU greater than a user configured IoU threshold (if provided by user) with ground truth 
    bounding boxes inside the frame(image). This function will calculate the normalized 
    (if is_normalize == True) offsets for all the positive anchor boxes as well as, it will also assign
    the categories to all the positive anchor boxes which will be the categories of ground truth bounding
    boxes.
    
    Parameters:
                iou (tensor) : IoU of each anchor box with each Ground Truth bounding box
                num_unique_categories (int) : Total number of categories in training data
                anchor_boxes_coords (tensor) : anchor boxes coordinates per feature map
                gt_info (tensor) : Ground truth bounding bounding box coordinates as well as class labels 
                                   of objects present in the image
                is_normalize (bool) : Whether to use normalization on the offsets calculated for positive
                                      anchor boxes or not.
                iou_threshold (float) : If this value is less than 1 then the function will go for the 
                                        second round to find out extra positive anchor boxes apart from 
                                        the ones which it has already found out which were having maximum 
                                        iou with the ground truth bounding boxes.
    Returning:
                positive_anchor_boxes_categories (tensor) : Tensor of categories which have been assigned 
                                                            to all the positive anchor boxes
                positive_anchor_boxes_offsets (tensor) : Normalized (if is_normalize == True) offsets for 
                                                         all the positive anchor boxes. 
                positive_anchor_boxes_indicator (tensor) : Tensor indicating which anchor boxes are marked
                                                           as positive anchor boxes. 
    """
    #Let's first try to find out all the positive anchor boxes which have the highest IoU with
    #ground truth bounding boxes among all the anchor boxes having different aspect ratios for a 
    #specific scale. 
    max_iou_anchor_boxes = np.argmax(iou,axis=0)
    
    #if the IoU threshold is less than 1 then we will go for second round of selection of credible positive
    #anchor boxes whicn have IoU with ground truth bounding boxes greater than the user selected IoU. 
    if iou_threshold < 1.0:
        secondary_pos_anchor_boxes = np.argwhere(iou > iou_threshold)
        
        #If we are getting some number of secondary positive anchor boxes then we have to determien their 
        #offsets as well as class labels (ground truth information). 
        if secondary_pos_anchor_boxes.size > 0:
            
            secondary_anchor_boxes = secondary_pos_anchor_boxes[:,0]
            secondary_anchor_boxes_categories = secondary_pos_anchor_boxes[:,1]
            
            secondary_anchor_boxes_gt_info = gt_info[secondary_anchor_boxes_categories]
            
            #Collecting all the primary as well as the secondary positive anchor boxes over an image
            all_positive_anchor_boxes = np.concatenate([max_iou_anchor_boxes,secondary_anchor_boxes],axis=0)
            
            #Collecting the ground truth bounding box coordinates as well as ground truth class labels for 
            #all the positive anchor boxes to calculate the offsets for all the positive anchor boxes. 
            all_positive_anchor_boxes_gt_info = np.concatenate([gt_info,secondary_anchor_boxes_gt_info],
                                                               axis=0)
            
        else:
            
            #If we didn't get any secondary positive anchor boxes while comparing with IoU threshold then
            # our all positive anchor boxes will be simply primary anchor boxes (max_iou_anchor_boxes)
            all_positive_anchor_boxes = max_iou_anchor_boxes
            
            #If we didn't get any secondary positive anchor boxes while comparing with IoU threshold then
            #the gt_info of all positive anchor boxes will be equal to the gt_info of all the primary anchor 
            #boxes
            all_positive_anchor_boxes_gt_info = gt_info
            
    else:
        
        all_positive_anchor_boxes = max_iou_anchor_boxes
        all_positive_anchor_boxes_gt_info = gt_info
        
    #The below tensor of positive anchor boxes indicator will be used during the calculation of Regression
    #loss while determining that which anchor boxes are positive anchor boxes because only for those
    #positive anchor boxes, the regression loss will be computed. 
    positive_anchor_boxes_indicator = np.zeros((iou.shape[0],4))
    
    #The tensor below is acting as a mask just like a boolean mask which will make the contribution of
    #all negative anchor boxes towards the regression loss (whether smooth L1 or L2 loss) zero and only 
    #all positive anchor boxes will be considered during computation because for them, there will be ones
    #at all the four places corresponsing to offsets. 
    positive_anchor_boxes_indicator[all_positive_anchor_boxes] = 1.0
        
    positive_anchor_boxes_categories = np.zeros((iou.shape[0],num_unique_categories))
    
    #Initially, all the anchor boxes for a specific scale will be assigned the category of background
    positive_anchor_boxes_categories[:,0] = 1
    
    #Then all the positive anchor boxes will be assigned to the specific object categories based on the 
    #category of that ground truth bounding box with which the IoU is maximum or greater than a specific
    #threshold.
    positive_anchor_boxes_categories[all_positive_anchor_boxes,0] = 0
    
    all_positive_anchor_boxes = all_positive_anchor_boxes.reshape(all_positive_anchor_boxes.shape[0],1)
    all_positive_anchor_boxes_categories = all_positive_anchor_boxes_gt_info[:,4].reshape(
                                            all_positive_anchor_boxes_gt_info.shape[0],1).astype(int)
    
    row_col = np.append(all_positive_anchor_boxes,all_positive_anchor_boxes_categories,axis=1)
    positive_anchor_boxes_categories[row_col[:,0],row_col[:,1]] = 1
    
    positive_anchor_boxes_offsets = np.zeros((iou.shape[0],4))
    
    if is_normalize == True:
        
        all_positive_anchor_boxes_gt_info = corner_to_center(all_positive_anchor_boxes_gt_info)
        anchor_boxes_coords = corner_to_center(anchor_boxes_coords)
        
        offsets_xy = all_positive_anchor_boxes_gt_info[:,0:2] - anchor_boxes_coords[all_positive_anchor_boxes[:,0],0:2]
        offsets_xy = offsets_xy/anchor_boxes_coords[all_positive_anchor_boxes[:,0],2:4]
        offsets_xy = offsets_xy/0.1
        
        offsets_wh = np.log(all_positive_anchor_boxes_gt_info[:,2:4]/anchor_boxes_coords[all_positive_anchor_boxes[:,0],2:4])
        offsets_wh = offsets_wh/0.2
        
        offsets = np.concatenate([offsets_xy,offsets_wh],axis=1)
    
    else:
        
        offsets = all_positive_anchor_boxes_gt_info[:,0:4] - anchor_boxes_coords[all_positive_anchor_boxes[:,0]]
        
    positive_anchor_boxes_offsets[all_positive_anchor_boxes[:,0]] = offsets
    
    return positive_anchor_boxes_categories,positive_anchor_boxes_offsets,positive_anchor_boxes_indicator


# In[7]:


def intersection(bboxes1,bboxes2):
    """This function will calculate the intersection between batches of two different boxes.
    
    Parameters:
                bboxes1 (tensor) : Batch of first bounding box coordinates
                bboxes2 (tensor) : Batch of second bounding box coordinates
                
    Returning:
                intersection_area (tensor) : intersection of areas bounded between batch of first bounding 
                                            box and second bounding box coordinates. 
    """
    #Let's try to find out that how many bounding boxes are there in the batch of first bounding boxes
    batch1 = bboxes1.shape[0]
    batch2 = bboxes2.shape[0]
    
    bboxes1_topmost_left_corner = np.expand_dims(bboxes1[:,[0,2]],axis=1)
    bboxes1_topmost_left_corner = np.tile(bboxes1_topmost_left_corner,reps=(1,batch2,1))
    
    bboxes2_topmost_left_corner = np.expand_dims(bboxes2[:,[0,2]],axis=0)
    bboxes2_topmost_left_corner = np.tile(bboxes2_topmost_left_corner,reps=(batch1,1,1))
    
    min_topmost_left = np.maximum(bboxes1_topmost_left_corner,bboxes2_topmost_left_corner)
    
    bboxes1_bottommost_right_corner = np.expand_dims(bboxes1[:,[1,3]],axis=1)
    bboxes1_bottommost_right_corner = np.tile(bboxes1_bottommost_right_corner,reps=(1,batch2,1))
    
    bboxes2_bottommost_right_corner = np.expand_dims(bboxes2[:,[1,3]],axis=0)
    bboxes2_bottommost_right_corner = np.tile(bboxes2_bottommost_right_corner,reps=(batch1,1,1))
    
    max_bottommost_right = np.minimum(bboxes1_bottommost_right_corner,bboxes2_bottommost_right_corner)
    
    wh = np.maximum(0,(max_bottommost_right - min_topmost_left))
    
    intersection_area = wh[:,:,0] * wh[:,:,1]
    
    return intersection_area


# In[8]:


def union(bboxes1,bboxes2):
    """This function will calculate the union areas between batches of two different boxes. 
    
     Parameters:
                bboxes1 (tensor) : Batch of first bounding box coordinates
                bboxes2 (tensor) : Batch of second bounding box coordinates
                
    Returning:
                union_area (tensor) : union of areas of batch of first bounding 
                                      box and second bounding box coordinates. 
    """
    #Let's try to find out that how many bounding boxes are there in the batch of first bounding boxes
    batch1 = bboxes1.shape[0]
    batch2 = bboxes2.shape[0]
    
    bboxes1_width = bboxes1[:,1] - bboxes1[:,0]
    bboxes1_height = bboxes1[:,3] - bboxes1[:,2]
    
    bboxes1_area = bboxes1_width * bboxes1_height
    bboxes1_area = np.expand_dims(bboxes1_area,axis=1)
    bboxes1_area = np.tile(bboxes1_area, reps=(1,batch2))
    
    bboxes2_width = bboxes2[:,1] - bboxes2[:,0]
    bboxes2_height = bboxes2[:,3] - bboxes2[:,2]
    
    bboxes2_area = bboxes2_width * bboxes2_height
    bboxes2_area = np.expand_dims(bboxes2_area,axis=0)
    bboxes2_area = np.tile(bboxes2_area, reps=(batch1,1))
    
    union_area = (bboxes1_area + bboxes2_area) - intersection(bboxes1,bboxes2)
    
    return union_area


# In[9]:


def compute_iou(bboxes1,bboxes2):
    """This function calculates the IoU between the batches of two different types of bounding boxes. 
     Parameters:
                bboxes1 (tensor) : Batch of first bounding box coordinates
                bboxes2 (tensor) : Batch of second bounding box coordinates
                
    Returning:
                iou_area (tensor) : Intersection over Union of areas of between batch of first and
                                    second group of bounding box coordinates. 
    """
    intersection_areas = intersection(bboxes1,bboxes2)
    union_areas = union(bboxes1,bboxes2)
    return intersection_areas/union_areas

