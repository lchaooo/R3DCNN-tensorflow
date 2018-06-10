# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange    # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
# from matplotlib import pyplot as plt
# mean_file='../crop_mean.npy'
# np_mean = np.load(mean_file)

def get_frames_data(filename, num_frames_per_clip=8, s_index = 0):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []
    # s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        if(len(filenames)<num_frames_per_clip):
            return [], s_index
        # if (s_index >= len(filenames)):
            # return [], s_index
        filenames = sorted(filenames)
        # s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = cv2.imread(image_name)
            # img_data = np.array(img)
            ret_arr.append(img)
    return ret_arr, s_index + num_frames_per_clip

def read_clip_and_label(lines, num_frames_per_clip=8, height=112, width=112, crop_center = False, shuffle = False):
    # lines = open(filename,'r')
    # read_dirnames = []
    data = []
    label = []
    # batch_index = 0
    # next_batch_start = -1
    # lines = list(lines)
    # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])  #(16, 112, 112, 3)
    # np_mean = 0
    # Forcing shuffle, if start_pos is not specified
    # if start_pos < 0:
        # shuffle = True
    # if shuffle:
        # video_indices = range(len(lines))
        # random.seed(time.time())
        # random.shuffle(video_indices)
    # else:
        # Process videos sequentially
        # video_indices = range(start_pos, len(lines))

    video_indices = range(len(lines))

    for index in video_indices:
        # if(batch_index>=batch_size):
            # next_batch_start = index
            # break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        # if not shuffle:
            # print("Loading a video clip from {}...".format(dirname))

        img_container = np.zeros((80 // num_frames_per_clip, num_frames_per_clip, height, width, 3), dtype=np.uint8)
        s_index = 0
        rotate = random.choice([True, True, False])
        scale = random.choice([True, True, False])
        angle = random.choice([-15, 0, 15])
	    scale_num = random.choice([0.8, 1.0, 1.2])
        if not crop_center:
            rand_seed=random.random()
            top = np.random.randint(0, 120 - height)
            left = np.random.randint(0, 160 - width)
        while s_index < 80:
            tmp_data, s_index = get_frames_data(dirname, num_frames_per_clip, s_index)
            for j in range(len(tmp_data)):
                img_cv = tmp_data[j]
		img = cv2.resize(img_cv, (160, 120))
		if not shuffle:
		    angle = 0
		if not shuffle:
		    scale_num = 1.0
		M = cv2.getRotationMatrix2D((80, 60), angle, 1.0)
		img = cv2.warpAffine(img, M, (160, 120))
        #if rotate & shuffle:
		#    if scale:
        #        M = cv2.getRotationMatrix2D((80, 60), angle, scale_num)
		#    else:
		#	M = cv2.getRotationMatrix2D((80, 60), angle, 1.0)
        #    img = cv2.warpAffine(img, M, (160, 120))
		#elif scale & shuffle:
		#    img = cv2.resize(img, (192, 144))
        if crop_center:
            img = cropCenter(np.array(img), height, width)
        else:
            img = RandomCrop(rand_seed, np.array(img), top, left, height=height, width=width)
        # plt.imshow(img)
        # plt.show()
                

        img_container[s_index // num_frames_per_clip - 1, j, :, :, :] = img[:,:,:]
        # img = np.array(cv2.resize(np.array(img), (128,171))).astype(np.float32)
        # img_datas.append(np.array(img)[:,:,0])
        # data.append(img_datas)
        # label.append(int(tmp_label))
        label.append(int(tmp_label))    
        # read_dirnames.append(dirname)
        data.extend([img_container])
        # batch_index = batch_index + 1
    # pad (duplicate) data/label if less than batch_size

    # valid_len = len(data)
    # pad_len = batch_size - valid_len
    # if pad_len:
    #     for i in range(pad_len):
    #         data.append(img_container)
    #         label.append(int(tmp_label))

    np_arr_data = np.array(data)
    np_arr_label = np.array(label)

    # return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len
    return np_arr_data, np_arr_label

def cropCenter(img, height, width):
    h,w,c = img.shape
    # print('h,w,c:', h,w,c)
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + width
    x1 = dx
    x2 = x1 + height
    # img = img[x1:x2,y1:y2,:]
    img = img[x1:x2,y1:y2,:]
    # print('img.shape:',img.shape)
    return img

def RandomCrop(rand_seed,img, top,left,height=112, width=112,u=0.5,aug_factor=9/8):
    #first zoom in by a factor of aug_factor of input img,then random crop by(height,width)
    # if rand_seed < u:
    if 1:
        # h,w,c = img.shape
        # img = cv2.resize(img, (round(aug_factor*w), round(aug_factor*h)), interpolation=cv2.INTER_LINEAR)
        # h, w, c = img.shape

        new_h, new_w = height,width

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
              left: left + new_w]
    
    return img

def randomHorizontalFlip(rand_seed,img, u=0.5):
    if rand_seed < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img

def normalize(arr):
    arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr

def sub_mean(batch_arr):
    # print('batch_arr.shape',batch_arr.shape)  #(32, 16, 112, 112, 3)
    for j in range(len(batch_arr)):
        batch_arr[j] -= np_mean
    return batch_arr

def train_aug(batch,is_train=True,Crop_heith=224,Crop_width=224,norm=True):
    new_batch=np.zeros((batch.shape[0],batch.shape[1],Crop_heith,Crop_width,3))
    # (16, 16, 112, 112, 3)
    rand_seed=random.random()
    random.seed(5)
    for i in range(batch.shape[0]):
        h, w, c = batch.shape[2:]
        new_h, new_w = Crop_heith, Crop_width
        dx = (h - Crop_heith) // 2
        dy = (w - Crop_width) // 2
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        for j in range(batch.shape[1]):
            if is_train:
                new_batch[i, j, :, :, :] = RandomCrop(rand_seed,batch[i, j, :, :, :],top,left,
                                                    height=Crop_heith, width=Crop_width)
                new_batch[i, j, :, :, :] = randomHorizontalFlip(rand_seed,new_batch[i, j, :, :, :])
            else:
                new_batch[i, j, :, :, :] = cv2.resize(batch[i, j, :, :, :],(Crop_width,Crop_heith))

    # return new_batch
    # return normalize(new_batch)
    if norm:
        return sub_mean(new_batch)
    else:
        return new_batch
