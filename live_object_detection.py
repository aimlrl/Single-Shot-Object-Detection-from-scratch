#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import argparse
import datetime
import skimage
from skimage.io import imread
from training_data_funcs import class2index, determine_anchor_box_color
import object_detection_config
from output_anchor_boxes_processing import display_predicted_boxes
from ssd_command_line_options import our_ssd_cnn_parser


# In[ ]:


#This is a hint for you to implement top_level_ssd, now
import top_level_ssd
from top_level_ssd import complete_ssd


# In[2]:


class  video_capture_demo():
    
    def __init__(self,detector,camera_index=0,width=500,height=375,is_record=False,
                 file_to_save="demo.mp4"):
        self.detector_obj = detector 
        self.camera_idx = camera_index
        self.video_width = width
        self.video_height = height
        self.record_or_not = is_record
        self.saving_file = file_to_save
        self.videowriter = None
        self.initialize_video_capture()
        

    def initialize_video_capture(self):
        self.video_capture = cv2.VideoCapture(self.camera_idx)
        
        if not self.video_capture.isOpened():
            print("Error opening video camera")
            return

        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)

        if self.record_or_not == True:
            self.videowriter = cv2.VideoWriter(self.saving_file,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,(self.video_width, self.video_height),isColor=True)

            
    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        pos = (10,30)
        font_scale = 0.9
        font_color = (0, 0, 0)
        line_type = 1

        while True:
            
            start_time = datetime.datetime.now()
            ret, image = self.video_capture.read()
            #cv2.imshow('image', image)
            
            img_to_detect = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255.0
            class_names,rects_to_draw = self.detector_obj.evaluate(input_image=img_to_detect)
            
            objects = {}
            for i in range(len(class_names)):
                
                rect = rects_to_draw[i]
                
                xmin = rect[0]
                ymin = rect[1]
                
                xmax = xmin + rect[2]
                ymax = ymin + rect[3]
                
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)
                
                name = class_names[i].split(":")[0]
                
                if name in objects.keys():
                    objects[name] += 1
                else:
                    objects[name] = 1
                    
                index = class2index(name)
                color = determine_anchor_box_color(index)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
                
                cv2.putText(image,name,(x1, y1-15),font,0.5,color,line_type)

            cv2.imshow('image', image)
            
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

            count = len(items.keys())
            
            if count > 0:
                
                x1 = 10
                y1 = 10
                x2 = 220
                y2 = 40 + count * 30
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

                total = 0.0
                
                for obj in objects.keys():
                    
                    count = objects[obj]
                    text_to_display = "%dx %s" % (count, obj)
                    
                    cv2.putText(image,text_to_display,(x1 + 10, y1 + 25),font,0.55,(0, 0, 0),
                                1)
                    y1 += 30

                cv2.line(image, (x1 + 10, y1), (x2 - 10, y1), (0,0,0), 1)

        self.capture.release()
        cv2.destroyAllWindows()


# In[ ]:


if __name__ == '__main__':
    
    parser = our_ssd_cnn_parser()
    
    desc = "Camera index"
    parser.add_argument("--camera_idx",default=0,type=int,help=desc)
    
    desc = "Whether to record video or not"
    parser.add_argument("--record_or_not",default=False,action='store_true',help=desc)
    
    desc = "File name to capture the video"
    parser.add_argument("--saving_file",default="demo.mp4",help=desc)

    args = parser.parse_args()
    
    #That's again a hint for you that how the complete_ssd class will be looking like
    ready_ssd = complete_ssd(args)
    
    if args.restore_weights == True:
        #That's again a hint for you that how restore_weights function might be looking like
        ready_ssd.restore_weights()
        
        videodemo = video_capture_demo(detector=ready_ssd,camera_index=args.camera_idx,
                                       is_record=args.record_or_not,file_to_save=args.saving_file)
        videodemo.loop()


# In[ ]:




