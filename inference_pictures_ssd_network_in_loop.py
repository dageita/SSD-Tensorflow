"""
Usage:
python /home/wangxf35/models-weights/tensorflow/SSD-Tensorflow/inference_pictures_ssd_network.py \
--data_dir=/path/to/folder \
--output_dir=/path/to/folder \
--ckpt_path=/path/to/model.ckpt \
--parallel_number=1 \
--select_threshold=0.16 \
--nms_threshold=0.3 \
--num_classes=2

TODO: 
1.The prediction time can be further improved by optimizing the parallel images processing part code.
2.Add read .pb function to inference.
"""

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization2
import time
from collections import namedtuple
from nets import nets_factory

class SSDNetInference(object):
    """  Importing and running isolated TF graph """
    def __init__(self,parallel_number,ckpt_path,select_threshold,nms_threshold,num_classes,model_name):
        self.parallel_number = parallel_number
        self.ckpt_path = ckpt_path
        self.select_threshold = select_threshold 
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes
        self.model_name = model_name
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        self.isess = tf.InteractiveSession(config=self.config)
        # Input placeholder.
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.images_list=[]
        self.images_input=tf.placeholder(tf.uint8, shape=(None, None, None, 3))
        for i in range(self.parallel_number):
            img_input = self.images_input[i,:]
            # Evaluation pre-processing: resize to SSD net shape.
            self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.images_list.append(tf.expand_dims(self.image_pre, 0))
        self.images_ssd=tf.concat(self.images_list,0)

        # Define the SSD model.
        self.reuse = True if 'ssd_net' in locals() else None
        self.ssd_class = nets_factory.get_network(self.model_name)
        self.ssd_params = self.ssd_class.default_params._replace(num_classes=self.num_classes)
        self.ssd_net = self.ssd_class(self.ssd_params)
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.images_ssd, is_training=False, reuse=self.reuse)

        # Restore SSD model.
        self.isess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.isess, self.ckpt_path)
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

    def run(self,data_dir,output_dir):
        # process images
        ret = 0
        toObjectDetection = None
        if os.path.isdir(data_dir):
            imgs=os.listdir(data_dir)
            imgs.sort()
        else:
            imgs=[data_dir[data_dir.rfind('/')+1:]]
            data_dir=data_dir[:data_dir.rfind('/')]
        i = 0
        parallel_number=self.parallel_number
        start_number= 0
        while start_number+parallel_number <= len(imgs):
            img = mpimg.imread(os.path.join(data_dir,imgs[start_number+i]))
            y = None
            for i in range(parallel_number):
                x = img[np.newaxis,:]
                if i >0:
                    y = np.concatenate((y,x),axis=0)
                else:
                    y = x
            # Run SSD network.
            start = time.time()
            rimg,rpredictions, rlocalisations,rbbox_img= self.isess.run([self.images_input,self.predictions, self.localisations,self.bbox_img],
                                                                    feed_dict={self.images_input: y})   
            end=time.time()
            print('prediction time ',end-start)  
            rpredictions_list=[]
            rlocalisations_list=[]
            for i in range(len(rpredictions)):
                rpredictions[i]=np.split(rpredictions[i], parallel_number, axis=0)
                rlocalisations[i]=np.split(rlocalisations[i], parallel_number, axis=0)
                rpredictions_list.append(np.array(rpredictions[i]))
                rlocalisations_list.append(np.array(rlocalisations[i]))
                
            result_list=[]
            for i in range(parallel_number):
                rpredictions=[]
                rlocalisations=[]
                for j in range(len(rpredictions_list)):
                    rpredictions.append(rpredictions_list[j][i])
                    rlocalisations.append(rlocalisations_list[j][i])
            # Get classes and bboxes from the net outputs.
                rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    rpredictions, rlocalisations, self.ssd_anchors,
                    select_threshold=self.select_threshold, img_shape=self.net_shape, num_classes=self.num_classes, decode=True)
                
                rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
                rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
                rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=self.nms_threshold)
                end = time.time()
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
                rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
                result_list.append([rclasses, rscores, rbboxes])   
            
            # visulization the images
            if not output_dir:
                output_dir=data_dir
            for i in range(parallel_number):
                [rclasses, rscores, rbboxes]=result_list[i]
                toObjectDetection=visualization2.bboxes_draw_on_img(img, rclasses, rscores, rbboxes,visualization2.colors_plasma,save_path=os.path.join(output_dir,imgs[start_number+i][:-4]+imgs[start_number+i][-4:]))
                # toObjectDetection = visualization2.plt_bboxes(img, rclasses, rscores, rbboxes,save_path=os.path.join(output_dir,imgs[start_number+i][:-4]+imgs[start_number+i][-4:]),figsize=(40,30))
            start_number+=parallel_number
        ret = 1
        return ret,toObjectDetection
# tf.app.run()

def main(parallel_number,ckpt_path,select_threshold,nms_threshold,num_classes,model_name):
    inference = SSDNetInference(parallel_number,ckpt_path,select_threshold,nms_threshold,num_classes,model_name)
    return inference
if __name__ == '__main__': 
    pass

