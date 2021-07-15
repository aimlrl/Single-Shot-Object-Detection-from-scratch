#!/usr/bin/env python
# coding: utf-8

# In[2]:


params = {
        'epoch_offset': 0,
        'classes' : ["Background","Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", 
                     "Motorbike","People","Table"]
        }


# In[3]:


def anchor_aspect_ratios():
    aspect_ratios = object_detection_config.params['aspect_ratios']
    return aspect_ratios

