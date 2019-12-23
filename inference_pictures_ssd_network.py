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
from notebooks import visualization
import time
from collections import namedtuple


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'parallel_number', 1,
    'The number of parallel inference pictures.')
tf.app.flags.DEFINE_string(
    'data_dir', './',
    'The folder of pictures to be inferenced.')
tf.app.flags.DEFINE_string(
    'output_dir', '../',
    'The folder of inferenced pictures to be storaged.')
tf.app.flags.DEFINE_string(
    'ckpt_path', './model.ckpt',
    'Path to model.ckpt')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.5,
    'The value of select bbox threshold.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45,
    'The value of nms threshold.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21,
    'number of classes includes background.')
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
images_list=[]
images_input=tf.placeholder(tf.uint8, shape=(None, None, None, 3))
for i in range(FLAGS.parallel_number):
    img_input = images_input[i,:]
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    images_list.append(tf.expand_dims(image_pre, 0))
images_ssd=tf.concat(images_list,0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])
ssdp = SSDParams(
                img_shape=(300, 300),
                num_classes=FLAGS.num_classes,
                no_annotation_label=2,
                feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
                feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                anchor_size_bounds=[0.15, 0.90],
                # anchor_size_bounds=[0.20, 0.90],
                anchor_sizes=[(21., 45.),
                            (45., 99.),
                            (99., 153.),
                            (153., 207.),
                            (207., 261.),
                            (261., 315.)],
                # anchor_sizes=[(30., 60.),
                #               (60., 111.),
                #               (111., 162.),
                #               (162., 213.),
                #               (213., 264.),
                #               (264., 315.)],
                anchor_ratios=[[2, .5],
                            [2, .5, 3, 1./3],
                            [2, .5, 3, 1./3],
                            [2, .5, 3, 1./3],
                            [2, .5],
                            [2, .5]],
                anchor_steps=[8, 16, 32, 64, 100, 300],
                anchor_offset=0.5,
                normalizations=[20, -1, -1, -1, -1, -1],
                prior_scaling=[0.1, 0.1, 0.2, 0.2]
                )
ssd_net = ssd_vgg_300.SSDNet(params = ssdp)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(images_ssd, is_training=False, reuse=reuse)

# Restore SSD model.
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, FLAGS.ckpt_path)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine
def process_images(imgs,select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300),parallel_number=4):
    # Run SSD network.
    start = time.time()
    rimg,rpredictions, rlocalisations,rbbox_img= isess.run([images_input,predictions, localisations,bbox_img],
                                                            feed_dict={images_input: imgs})                                                        
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
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=FLAGS.num_classes, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

        end = time.time()
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        result_list.append([rclasses, rscores, rbboxes])        
    return result_list

def main(_):
    imgs=os.listdir(FLAGS.data_dir)
    imgs.sort()
    i = 0
    parallel_number=FLAGS.parallel_number
    start_number= 0
    while start_number+parallel_number <= len(imgs):
        img = mpimg.imread(os.path.join(FLAGS.data_dir,imgs[start_number+i]))
        y = None
        for i in range(parallel_number):
            x = img[np.newaxis,:]
            if i >0:
                y = np.concatenate((y,x),axis=0)
            else:
                y = x
        result_list = process_images(y,select_threshold=FLAGS.select_threshold,nms_threshold=FLAGS.nms_threshold,parallel_number=parallel_number)
        for i in range(parallel_number):
            [rclasses, rscores, rbboxes]=result_list[i]
            visualization.plt_bboxes(img, rclasses, rscores, rbboxes,save_path=os.path.join(FLAGS.output_dir,imgs[start_number+i][:-4]+'_infer'+imgs[start_number+i][-4:]))
        start_number+=parallel_number

if __name__ == '__main__':
    tf.app.run()
