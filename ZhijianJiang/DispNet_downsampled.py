# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import time
from datetime import date
from datetime import datetime
import pickle as cPickle
#import _pickle as cPickle

##from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
#from tensorflow.core.protobuf import saver_pb2
import matplotlib.pyplot as plt
#import webp

TEMPDIR = 'C://Users//PSIML7//Desktop//Stereo2Depth//CNNs//Kinez//DispNet-TensorFlow-master'
IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384
BATCH_SIZE = 10
#TOTAL_IMAGES = 12
#BATCH_NUM = TOTAL_IMAGES // BATCH_SIZE
#ROUND_STEP = 12
#EPOCH = 30
EPOCH = 2
LEARNING_RATE =1e-5
#MODELS_PATH = "/home/visg1/jzj/models/model.ckpt.data-00000-of-00001"
MODELS_DIR = './model'
DATA_DIR = './data_fin'
LOGS_DIR = './logs'
RUNNING_LOGS_DIR = './running_logs'
#OUTPUT_DIR = './output'
OUTPUT_DIR = './FlyingThings3d/output'
GT_DIR = './test/outputs/gopro2.pkl'
TRAIN_SERIES = list(range(132)) + list(range(136, 140)) + list(range(144, 150)) + list(range(160, 198)) + list(range(202, 220)) + list(range(224, 230)) + list(range(234, 248)) + list(range(258, 292)) + list(range(296, 302)) + list(range(306, 324))
image_num = np.size(TRAIN_SERIES)
SAVE_PER_EPOCH = 1
SAVE_PER_BATCH = 10
VALIDATION_SIZE = 100
VALIDATION_BATCHES = VALIDATION_SIZE//BATCH_SIZE

#crop_up = 78
#crop_down = 462
#crop_left = 96
#crop_right = 864
crop_up = 174
crop_down = 366
crop_left = 288
crop_right = 672
IMAGE_SIZE_X = crop_right-crop_left
IMAGE_SIZE_Y = crop_down-crop_up

timings = {
    'sum':0,
    'count':-1,
    'last':0
}
def trackTime(moment, timer = timings):
    timer['count'] +=1
    temp_time = moment
    if timer['count'] > 0:
        last_duration = temp_time - timer['last']
        timer['sum'] += last_duration
        print("last batch duration: {}, average batch duration: {}, total number of batches: {}".format(last_duration, timer['sum']/timer['count'], timer['count']))
    timer['last'] = temp_time
    #file.write("last batch duration: {}, average batch duration: {}, total number of batches: {}".format(last_duration, timer['sum']/timer['count'], timer['count']))
    return(timer)

def py_avg_pool(value, strides):
	batch_size, height, width, channel_size = value.shape
	res_height = int(height / strides[1])
	res_width = int(width / strides[2])
	print(res_height, res_width)
	result = np.zeros((batch_size, res_height, res_width, 1))
	for i in range(res_height):
		for j in range(res_width):
			for k in range(batch_size):
				result[k, i, j, 0] = np.mean(value[k, i * int(strides[1]) : (i + 1) * int(strides[1]), j * int(strides[2]) : (j + 1) * int(strides[2]), :])
	return result



def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name='weight')
    #return tf.Variable(initial, name='W')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')

def conv2d(x, W, strides):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def upconv2d_2x2(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def upconv2d_1x1(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME');


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

def loss(pre, gt):
    loss = tf.sqrt(tf.reduce_mean(tf.square(pre - gt)))
    return loss

def pre(conv):
    return tf.expand_dims(tf.reduce_mean(conv, 3), -1)
    #return tf.reduce_mean(conv, 3)

def model(combine_image, ground_truth):
  # conv1
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([7,7, 6,64]) 
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(combine_image, W_conv1, [1, 2, 2 ,1]) + b_conv1) 
    #h_pool1 = max_pool_2x2(h_conv1)   
    # h_pool1 = h_conv1 

  # conv2
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5,5, 64,128]) 
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [ 1, 2, 2, 1]) + b_conv2) 
    #h_pool2 = max_pool_2x2(h_conv2)   
    h_pool2 = h_conv2                                      
  # conv3a
  with tf.name_scope('conv3a'):
    W_conv3a = weight_variable([5,5, 128,256]) 
    b_conv3a = bias_variable([256])
    h_conv3a = tf.nn.relu(conv2d(h_conv2, W_conv3a, [1, 2, 2, 1]) + b_conv3a) 
    #h_pool3a = max_pool_2x2(h_conv3a)
    #h_pool3a = h_conv3a

  # conv3b
  with tf.name_scope('conv3b'):
    W_conv3b = weight_variable([3,3, 256,256]) 
    b_conv3b = bias_variable([256])
    h_conv3b = tf.nn.relu(conv2d(h_conv3a, W_conv3b, [1, 1, 1, 1]) + b_conv3b) 
    #h_pool3b = h_conv3b

  # conv4a
  with tf.name_scope('conv4a'):
    W_conv4a = weight_variable([3,3, 256,512]) 
    b_conv4a = bias_variable([512])
    h_conv4a = tf.nn.relu(conv2d(h_conv3b, W_conv4a, [1, 2, 2, 1]) + b_conv4a) 
    #h_pool4a = max_pool_2x2(h_conv4a) 
    #h_pool4a = h_conv4a
  # conv4b
  with tf.name_scope('conv4b'):
    W_conv4b = weight_variable([3,3, 512,512]) 
    b_conv4b = bias_variable([512])
    h_conv4b = tf.nn.relu(conv2d(h_conv4a, W_conv4b, [1, 1, 1, 1]) + b_conv4b) 
    #h_pool4b = h_conv4b

  # conv5a
  with tf.name_scope('conv5a'):
    W_conv5a = weight_variable([3,3, 512,512]) 
    b_conv5a = bias_variable([512])
    h_conv5a = tf.nn.relu(conv2d(h_conv4b, W_conv5a, [1, 2, 2, 1]) + b_conv5a) 
    #h_pool5a = max_pool_2x2(h_conv5a) 
    #h_pool5a = h_conv5a
  # conv5b
  with tf.name_scope('conv5b'):
    W_conv5b = weight_variable([3,3, 512,512]) 
    b_conv5b = bias_variable([512])
    h_conv5b = tf.nn.relu(conv2d(h_conv5a, W_conv5b, [ 1, 1, 1, 1]) + b_conv5b) 
    #h_pool5b = h_conv5b

  # conv6a
  with tf.name_scope('conv6a'):
    W_conv6a = weight_variable([3,3, 512,1024]) 
    b_conv6a = bias_variable([1024])
    h_conv6a = tf.nn.relu(conv2d(h_conv5b, W_conv6a, [1, 2, 2, 1]) + b_conv6a) 
    #h_pool6a = max_pool_2x2(h_conv6a) 
    #h_pool6a = h_conv6a
  # conv6b
  with tf.name_scope('conv6b'):
    W_conv6b = weight_variable([3,3, 1024,1024]) 
    b_conv6b = bias_variable([1024])
    h_conv6b = tf.nn.relu(conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b) 
    #h_pool6b = h_conv6b

  # pr6 + loss6
  with tf.name_scope('pr6_loss6'):
    W_pr6 = weight_variable([3,3, 1024,1]) 
    b_pr6 = bias_variable([1])
    pr6 = tf.nn.relu(conv2d(h_conv6b, W_pr6, [1, 1, 1, 1]) + b_pr6)
    # h_conv6b = tf.nn.relu(conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b)
    # pr6 = pre(h_conv6b)
    gt6 = tf.nn.avg_pool(ground_truth, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6')
    loss6 = loss(pr6, gt6)

  # upconv5
  with tf.name_scope('upconv5'):
    W_upconv5 = weight_variable([4,4, 512,1024]) 
    b_upconv5 = bias_variable([512])
    h_upconv5 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_conv6b,  W_upconv5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 32), np.int32(IMAGE_SIZE_X / 32), 512]) + b_upconv5, center=True, scale=True, is_training=True)) 


  # iconv5
  with tf.name_scope('iconv5'):
    W_iconv5 = weight_variable([3,3, 1024,512]) 
    b_iconv5 = bias_variable([512])
    h_iconv5 = tf.nn.relu(conv2d(tf.concat([h_upconv5, h_conv5b], 3), W_iconv5, [1, 1, 1, 1]) + b_iconv5) 


  # pr5 + loss5
  with tf.name_scope('pr5_loss5'):
    W_pr5 = weight_variable([3,3, 512,1]) 
    b_pr5 = bias_variable([1])
    pr5 = tf.nn.relu(conv2d(h_iconv5, W_pr5, [1, 1, 1, 1]) + b_pr5)    
    # pr5 = pre(h_iconv5)
    gt5 = tf.nn.avg_pool(ground_truth, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5')
    loss5 = loss(pr5, gt5)

  # upconv4
  with tf.name_scope('upconv4'):
    W_upconv4 = weight_variable([4,4, 256, 512])
    b_upconv4 = bias_variable([256])
    h_upconv4 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 16), np.int32(IMAGE_SIZE_X / 16), 256]) + b_upconv4, center=True, scale=True, is_training=True))


  # iconv4
  with tf.name_scope('iconv4'):
    W_iconv4 = weight_variable([3,3, 768,256]) 
    b_iconv4 = bias_variable([256])
    h_iconv4 = tf.nn.relu(conv2d(tf.concat([h_upconv4, h_conv4b], 3), W_iconv4, [ 1, 1, 1, 1]) + b_iconv4) 


  # pr4 + loss4
  with tf.name_scope('pr4_loss4'):
    W_pr4 = weight_variable([3,3, 256,1]) 
    b_pr4 = bias_variable([1])
    pr4 = tf.nn.relu(conv2d(h_iconv4, W_pr4, [1, 1, 1, 1]) + b_pr4)    
    # pr4 = pre(h_iconv4)
    gt4 = tf.nn.avg_pool(ground_truth, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
    loss4 = loss(pr4, gt4)

  # upconv3
  with tf.name_scope('upconv3'):
    W_upconv3 = weight_variable([4,4,128, 256]) 
    b_upconv3 = bias_variable([128])
    h_upconv3 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv4, W_upconv3, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 8), np.int32(IMAGE_SIZE_X / 8), 128]) + b_upconv3, center=True, scale=True, is_training=True)) 


  # iconv3
  with tf.name_scope('iconv3'):
    W_iconv3 = weight_variable([3,3, 384,128]) 
    b_iconv3 = bias_variable([128])
    h_iconv3 = tf.nn.relu(conv2d(tf.concat([h_upconv3, h_conv3b], 3), W_iconv3, [ 1, 1, 1, 1]) + b_iconv3) 


  # pr3 + loss3
  with tf.name_scope('pr3_loss3'):
    W_pr3 = weight_variable([3,3, 128,1]) 
    b_pr3 = bias_variable([1])
    pr3 = tf.nn.relu(conv2d(h_iconv3, W_pr3, [1, 1, 1, 1]) + b_pr3) 
    # pr3 = pre(h_iconv3)
    gt3 = tf.nn.avg_pool(ground_truth, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
    loss3 = loss(pr3, gt3)

  # upconv2
  with tf.name_scope('upconv2'):
    W_upconv2 = weight_variable([4,4,64, 128]) 
    b_upconv2 = bias_variable([64])
    h_upconv2 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 4), np.int32(IMAGE_SIZE_X / 4), 64]) + b_upconv2, center=True, scale=True, is_training=True)) 


  # iconv2
  with tf.name_scope('iconv2'):
    W_iconv2 = weight_variable([3,3, 192,64]) 
    b_iconv2 = bias_variable([64])
    h_iconv2 = tf.nn.relu(conv2d(tf.concat([h_upconv2, h_conv2], 3), W_iconv2, [1, 1, 1, 1]) + b_iconv2) 


  # pr2 + loss2
  with tf.name_scope('pr2_loss2'):
    W_pr2 = weight_variable([3,3, 64,1]) 
    b_pr2 = bias_variable([1])
    pr2 = tf.nn.relu(conv2d(h_iconv2, W_pr2, [1, 1, 1, 1]) + b_pr2) 
    # pr2 = pre(h_iconv2)
    gt2 = tf.nn.avg_pool(ground_truth, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
    loss2 = loss(pr2, gt2)

  # upconv1
  with tf.name_scope('upconv1'):
    W_upconv1 = weight_variable([4,4,32, 64]) 
    b_upconv1 = bias_variable([32])
    h_upconv1 = tf.nn.relu(tf.contrib.layers.batch_norm(upconv2d_2x2(h_iconv2, W_upconv1, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y / 2), np.int32(IMAGE_SIZE_X / 2), 32]) + b_upconv1, center=True, scale=True, is_training=True)) 

  # iconv1
  with tf.name_scope('iconv1'):
    W_iconv1 = weight_variable([3,3, 96,32]) 
    b_iconv1 = bias_variable([32])
    h_iconv1 = tf.nn.relu(conv2d(tf.concat([h_upconv1, h_conv1], 3), W_iconv1, [ 1, 1, 1, 1]) + b_iconv1) 

  # pr1 + loss1
  with tf.name_scope('pr1_loss1'):
    W_pr1 = weight_variable([3,3, 32,1]) 
    b_pr1 = bias_variable([1])
    pr1 = tf.nn.relu(conv2d(h_iconv1, W_pr1, [1, 1, 1, 1]) + b_pr1) 
    # pr1 = pre(h_iconv1)
    gt1 = tf.nn.avg_pool(ground_truth, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
    loss1 = loss(pr1, gt1)
  

  final_output = pr1
  # overall loss
  with tf.name_scope('loss'):
    total_loss = ( 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6)
  return final_output, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, tf.is_inf(loss6)

def _norm(img):
  return (img - np.mean(img)) / np.std(img)

def load_pfm(fname, crop = False):
    if crop:
        if not os.path.isfile(fname + '.H.pfm'):
            x, scale = load_pfm(fname, False)
            x_ = np.zeros((384, 768), dtype=np.float32)
            for i in range(77, 461):
                for j in range(96, 864):
                    x_[i - 77, j - 96] = x[i, j]
            save_pfm(fname + '.H.pfm', x_, scale)
            return x_, scale
        else:
            fname += '.H.pfm'
    color = None
    width = None
    height = None
    scale = None
    endian = None
  
    file = open(fname, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True    
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
 
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
 
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
 
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale

def load_pfm_files():
    output_dir = 'C://Users//PSIML7//Desktop//Stereo2Depth//CNNs//Kinez//DispNet-TensorFlow-master//test//outputs'
    assert(os.path.isdir(output_dir))
    dispnoc = []
    base1 = 'C://Users//PSIML7//Desktop//Stereo2Depth//CNNs//Kinez//DispNet-TensorFlow-master//test//inputs//'

    filenames = sorted(os.listdir(base1))

    for i in range(np.size(filenames)):
	    disp, scale = load_pfm(''.join([base1, filenames[i]]), False)
	    dispnoc.append(disp.astype(np.float32))
    return dispnoc


def loadAllPaths(datasetp,subfolder = ['A','B','C'],lr = True):
    retvall = []
    retvalr = []
    retval = []
    for subf in subfolder:
        for scene in os.listdir(os.path.join(datasetp,subf)):
            if lr:
                for fajl in os.listdir(os.path.join(datasetp,subf,scene,'left')):
                    retvall.append(os.path.join(datasetp,subf,scene,'left',fajl))
                    retvalr.append(os.path.join(datasetp,subf,scene,'right',fajl))
            else: # not lr
                for fajl in os.listdir(os.path.join(datasetp,subf,scene)):
                    retval.append(os.path.join(datasetp,subf,scene,fajl))
    retval = np.array(retval)
    retvalr = np.array(retvalr)
    retvall = np.array(retvall)
    return(retval,retvalr,retvall)


    



def main():

  training_timestamp = str(datetime.now()).replace(":","-").replace(" ","_").replace(".","-")[:19]
  os.mkdir(os.path.join(OUTPUT_DIR,training_timestamp))
  file1 = open(os.path.join(OUTPUT_DIR, training_timestamp, "log.txt"),"w+")
  print(1)

  _,input_right,input_left = loadAllPaths(datasetp = os.path.join(TEMPDIR,"./FlyingThings3d/input_webp/frames_cleanpass_webp/TRAIN"),subfolder = ['A','B','C'], lr = True)
  _,output_right,output_left = loadAllPaths(datasetp = os.path.join(TEMPDIR,"./FlyingThings3d/disparity/disparity/TRAIN"),subfolder = ['A','B','C'], lr = True)
  TOTAL_IMAGES = input_right.shape[0]

  printing_left = input_left[TOTAL_IMAGES-1]
  printing_right = input_right[TOTAL_IMAGES-1]
  printing_output = output_left[TOTAL_IMAGES-1]
  TOTAL_IMAGES = TOTAL_IMAGES-1

  input_one_image = Image.open(printing_left).convert('RGB')
  in_arr = np.array(input_one_image)
  in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]  
  printing_left = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
  printing_lefts = printing_left
  for i in range(1,10):
      printing_lefts = np.concatenate((printing_lefts, printing_left), axis = 0)


  input_one_image = Image.open(printing_right).convert('RGB')
  in_arr = np.array(input_one_image)
  in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]  
  printing_right = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
  printing_rights = printing_right
  for i in range(1,10):
      printing_rights = np.concatenate((printing_rights, printing_right), axis = 0)

  input_one_image = load_pfm(printing_output)[0]
  in_arr = np.array(input_one_image)
  in_arr = in_arr[crop_up:crop_down,crop_left:crop_right]  
  input_one_image = np.reshape(in_arr, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
  printing_output = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1)) 
  printing_outputs = printing_output
  for i in range(1,10):
      printing_outputs = np.concatenate((printing_outputs, printing_output), axis = 0)

  

  validation_left = input_left[TOTAL_IMAGES-VALIDATION_SIZE:]
  validation_right = input_right[TOTAL_IMAGES-VALIDATION_SIZE:]
  validation_output = output_left[TOTAL_IMAGES-VALIDATION_SIZE:]
  TOTAL_IMAGES = TOTAL_IMAGES-VALIDATION_SIZE
  
  BATCH_NUM = TOTAL_IMAGES // BATCH_SIZE
  BATCH_NUM = 20
  VALIDATION_BATCHES = 2

  print("TOTAL IMAGES : {}, BATCH_NUM: {} ".format(TOTAL_IMAGES,BATCH_NUM))
  file1.write("TOTAL IMAGES : {}, BATCH_NUM: {} \n".format(TOTAL_IMAGES,BATCH_NUM))

 # with open(GT_DIR) as f:
  #    buf = cPickle.load(f, encoding="utf-8")
  #buf = load_pfm_files()
  
  image_left = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_left')
  image_right = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_right')
  ground_truth = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='ground_truth')
  #image_left = tf.placeholder(tf.float32, [None, crop_down-crop_up, crop_right-crop_left, 3], name='image_left')
  #image_right = tf.placeholder(tf.float32, [None, crop_down-crop_up, crop_right-crop_left, 3], name='image_right')
  #ground_truth = tf.placeholder(tf.float32, [None,  crop_down-crop_up, crop_right-crop_left, 1], name='ground_truth')
  

  is_training = tf.placeholder(tf.bool, name = "is_training")
  #Maksim je pica!
  combine_image = tf.concat([image_left, image_right], 3)
  final_output, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf = model(combine_image=combine_image, 
                            ground_truth=ground_truth)
  tf.summary.scalar('loss', total_loss)

  with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)
 
  merged = tf.summary.merge_all()

  # important step
  sess = tf.Session()
  
  if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
      writer = tf.train.SummaryWriter(LOGS_DIR, sess.graph)
  else: # tensorflow version >= 0.12
      writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)


  #left_images = sorted(os.listdir(DATA_DIR+ '//left//'))
  #right_images = sorted(os.listdir(DATA_DIR + '//right//'))
  left_images = input_left
  right_images = input_right
  # output_images = sorted(os.listdir(DATA_DIR + '/output/'))  


  # tf.initialize_all_variables() no long valid from
  # 2017-03-02 if using tensorflow >= 0.12

  if int((tf.__version__).split('.')[1]) < 12:
      init = tf.initialize_all_variables()
  else:
      init = tf.global_variables_initializer()
  # saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
  saver = tf.train.Saver()
   
  
  sess.run(init)
  # saver.restore(sess, MODEL_PATH)
  with open(RUNNING_LOGS_DIR + "/log" + date.isoformat(date.today()) + str(time.time()) + ".txt", "w+") as file:
 #   file.write('BATCH_SIZE ' + str(BATCH_SIZE) + '\n'
	#+ ' EPOCH ' + str(EPOCH) + '\n'
	# + ' image_num ' + str(image_num) + '\n' 
	#+ ' LEARNING_RATE ' + str(LEARNING_RATE) + '\n')

    for round in range(EPOCH):
      #for i in range(0 , image_num - BATCH_SIZE, ROUND_STEP):
      for i in range(BATCH_NUM):

        trackTime(time.time())
        for j in range(BATCH_SIZE):
          if (i == BATCH_NUM-1 and i*BATCH_SIZE+j >= TOTAL_IMAGES ):
            break
          

          # input data
          #full_pic_name = DATA_DIR+ '/left/' + left_images[TRAIN_SERIES[i + j]]
          full_pic_name = left_images[i*BATCH_SIZE+j]
          input_one_image = Image.open(full_pic_name).convert('RGB')
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]  
          input_one_image = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_left_images = input_one_image
          else:
            input_left_images = np.concatenate((input_left_images, input_one_image), axis=0)

          #full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
          full_pic_name = right_images[i*BATCH_SIZE+j]
          input_one_image = Image.open(full_pic_name).convert('RGB')
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]
          input_one_image = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_right_images = input_one_image
          else:
            input_right_images = np.concatenate((input_right_images, input_one_image), axis=0)

          #input_one_image = buf[TRAIN_SERIES[i + j]]
          input_one_image = load_pfm(output_left[i*BATCH_SIZE+j])[0]
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right]  
          input_one_image = np.reshape(in_arr, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1)) 

          if(j == 0):
  	        input_gts = input_one_image
          else:
            input_gts = np.concatenate((input_gts, input_one_image), axis=0)

        result, optimizer_res, total_loss_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res, loss6_inf_res =sess.run([merged, optimizer, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf],feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
        #result, optimizer_res, total_loss_res =sess.run([merged, optimizer, total_loss],feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
	    
        print("training: epoch: {}, batch num: {}, total loss: {}".format(round,i,total_loss_res))
        print("training: losses: total={}, {},{},{},{},{},{}".format(total_loss_res,loss1_res,loss2_res,loss3_res,loss4_res,loss5_res,loss6_res ))
        file1.write("training: epoch: {}, batch num: {}, total loss: {} \n".format(round,i,total_loss_res))
        file1.write("training: losses: total={}, {},{},{},{},{},{} \n".format(total_loss_res,loss1_res,loss2_res,loss3_res,loss4_res,loss5_res,loss6_res ))
        #print("losses:" + str(total_loss_res) + " " + str(loss1_res) + " " + str(loss2_res) + " " + str(loss3_res) + " " + str(loss4_res) + " " + str(loss5_res) + " " + str(loss6_res))
        #print("round: "str(round) + " total loss: " + str(total_loss_res))
        #print("losses:" + str(total_loss_res) + " " + str(loss1_res) + " " + str(loss2_res) + " " + str(loss3_res) + " " + str(loss4_res) + " " + str(loss5_res) + " " + str(loss6_res))

        if round%SAVE_PER_EPOCH == SAVE_PER_EPOCH-1 and i%SAVE_PER_BATCH == SAVE_PER_BATCH-1:
            #print("ajde bre sacuvaj")
            #result, total_loss_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res, loss6_inf_res =sess.run([merged, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf],feed_dict={image_left:printing_left, image_right:printing_right, ground_truth:printing_output})
            result, total_loss_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res, loss6_inf_res =sess.run([merged, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf],feed_dict={image_left:printing_lefts, image_right:printing_rights, ground_truth:printing_outputs})
       
            #slika = slika 
            #slika = np.min(1,slika)
            plt.imsave(os.path.join(OUTPUT_DIR,training_timestamp,"epoch{}_batch{}.png".format(round,i)), pr1_res[0,:,:,0], cmap="gray")

            ##plt.imshow(slika[:,:,0], cmap="gray")
            ##plt.show()

      saver = tf.train.Saver()
      saver.save(sess, os.path.join(OUTPUT_DIR,training_timestamp,"model_epoch{}.ckpt".format(round)))


      for i in range(VALIDATION_BATCHES):
        trackTime(time.time())
        for j in range(BATCH_SIZE):
          if (i == BATCH_NUM-1 and i*BATCH_SIZE+j >= VALIDATION_SIZE ):
            break
          

          # input data
          #full_pic_name = DATA_DIR+ '/left/' + left_images[TRAIN_SERIES[i + j]]
          full_pic_name = validation_left[i*BATCH_SIZE+j]
          input_one_image = Image.open(full_pic_name).convert('RGB')
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]  
          input_one_image = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_left_images = input_one_image
          else:
            input_left_images = np.concatenate((input_left_images, input_one_image), axis=0)

          #full_pic_name = DATA_DIR + '/right/' + right_images[TRAIN_SERIES[i + j]]
          full_pic_name = validation_right[i*BATCH_SIZE+j]
          input_one_image = Image.open(full_pic_name).convert('RGB')
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right,:]
          input_one_image = _norm(np.reshape(in_arr, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
          if(j == 0):
  	        input_right_images = input_one_image
          else:
            input_right_images = np.concatenate((input_right_images, input_one_image), axis=0)

          #input_one_image = buf[TRAIN_SERIES[i + j]]
          input_one_image = load_pfm(validation_output[i*BATCH_SIZE+j])[0]
          in_arr = np.array(input_one_image)
          in_arr = in_arr[crop_up:crop_down,crop_left:crop_right]  
          input_one_image = np.reshape(in_arr, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))

          if(j == 0):
  	        input_gts = input_one_image
          else:
            input_gts = np.concatenate((input_gts, input_one_image), axis=0)

        result, total_loss_res, loss1_res, loss2_res, loss3_res, loss4_res, loss5_res, loss6_res, pr6_res, pr5_res, pr4_res, pr3_res, pr2_res, pr1_res, loss6_inf_res =sess.run([merged, total_loss, loss1, loss2, loss3, loss4, loss5, loss6, pr6, pr5, pr4, pr3, pr2, pr1, loss6_inf],feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
        #result, optimizer_res, total_loss_res =sess.run([merged, optimizer, total_loss],feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
	    
        print("validation: epoch: {}, batch num: {}, total loss: {}".format(round,i,total_loss_res)) 
        print("validation: losses: total={}, {},{},{},{},{},{}".format(total_loss_res,loss1_res,loss2_res,loss3_res,loss4_res,loss5_res,loss6_res ))
        file1.write("validation: epoch: {}, batch num: {}, total loss: {} \n".format(round,i,total_loss_res)) 
        file1.write("validation: losses: total={}, {},{},{},{},{},{} \n".format(total_loss_res,loss1_res,loss2_res,loss3_res,loss4_res,loss5_res,loss6_res ))
        
        
        
        #if round == EPOCH - 1:
        #  final_result = (sess.run(final_output, feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})) 
        #  for k in range(1):
        #    result = np.squeeze(final_result[k])
        #    result = result.astype(np.uint8)
        #    #plt.imsave(OUTPUT_DIR + '/' + str(i) + '.png', result, format='png')
        #    file1.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(total_loss_res) +' loss1 ' + str(loss1_res) +  '\n')

        #if i == 0:
        #    #print(' pr1_real_loss ' + str(np.sqrt(np.mean(np.square(py_avg_pool(input_gts, [1,2,2,1]), pr1_res)))))
        #    file1.write('round ' + str(round) + ' batch ' + str(i) + ' total_loss ' + str(total_loss_res) +' loss1 ' + str(loss1_res) +  '\n')

if __name__ == '__main__':
  main()
