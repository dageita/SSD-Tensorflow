"""
Usage:
python /home/lenovo/wangxf35/dageita/SSD-Tensorflow/inference_realtime_camera_videos_ssd_network.py \
--rtsp='rtsp://admin:lenovo123@192.168.1.120:554'
--output_path=/home/wangxf35/data/pre_photo_post_check/internal_detection/test_result/test.avi \
--ckpt_path=/home/lenovo/wangxf35/test/model.ckpt-8691 \
--select_threshold=0.22 \
--nms_threshold=0.15 \
--num_classes=5 \
--video_frame_rate=25 \
--video_resolution=4000,3000

"""

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import queue,threading
q=queue.Queue()

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import time


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'rtsp', 'rtsp://admin:lenovo123@192.168.1.120:554',
    'rtsp url')
tf.app.flags.DEFINE_string(
    'output_path', '../',
    'The path of inferenced realtime video to be storaged, end with xxx.avi')
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
tf.app.flags.DEFINE_integer(
    'video_frame_rate', 25,
    'video frame rate.')
tf.app.flags.DEFINE_list(
    'video_resolution', [3000,4000],
    'video resolution.')

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
images_list=[]
images_input=tf.placeholder(tf.uint8, shape=(None, None, None, 3))
for i in range(1):
    img_input = images_input[i,:]
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    images_list.append(tf.expand_dims(image_pre, 0))
images_ssd=tf.concat(images_list,0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_class = nets_factory.get_network(FLAGS.model_name)
ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
ssd_net = ssd_class(ssd_params)
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
    for i in range(1):
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



def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture(FLAGS.rtsp)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
    cap.release()

def Process():
    out = cv2.VideoWriter(FLAGS.output_path, cv2.VideoWriter_fourcc(*'XVID'), FLAGS.video_frame_rate, FLAGS.video_resolution[0], FLAGS.video_resolution[1]))
    print("Start Displaying")
    number = 0
    while True:
        if q.empty() !=True:
            frame=q.get()
            x = frame[np.newaxis,:]
            number+=1
            result_list=process_images(x,select_threshold=FLAGS.select_threshold,nms_threshold=FLAGS.nms_threshold,parallel_number=1)
            [rclasses, rscores, rbboxes]=result_list[i]
            output_rgb=visualization.bboxes_draw_on_img(frame, rclasses, rscores, rbboxes,visualization.colors_plasma)
            out.write(output_rgb)
            print("number is",number)
            number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            break


def main(_):
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Process)
    p1.start()
    p2.start()

if __name__ == '__main__':
    tf.app.run()
