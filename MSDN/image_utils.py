# ---------------------------------------------------------
# Image and Dataset Util Implementation
# Licensed under The KIST License
# Written by CSRC, KIST
# ---------------------------------------------------------
import natsort
import numpy as np
import os
from PIL import Image


class image_utils(object):
    def __init__(self, path, w=224, h=224, c=3, c_channel=None):
        self.path = path
        self.width = w
        self.height = h
        self.channel = c
        self.c_channel = c_channel
        self.properties = []

    def load_data(self, shuffle_list=None):
        path_list = os.listdir(self.path)
        path_list.sort()
        path_array = []
        s_path_array = []

        if self.c_channel is None:
            print("Please assign the axis (x, y and z) for c_channel")
            exit(1)
        elif self.c_channel is 'x':
            for class_name in path_list:
                directFiles = os.listdir(self.path + "/" + class_name + "/" + self.c_channel + "/")
                files = natsort.natsorted(directFiles)
                for file in files:
                    jpg = self.path + "/" + class_name + "/" + self.c_channel + "/" + file
                    path_array.append(jpg)
        else:
            for class_name in path_list:
                directFiles = os.listdir(self.path + "/" + class_name + "/" + self.c_channel + "/")
                files = natsort.natsorted(directFiles)
                for file in files:
                    jpg = self.path + "/" + class_name + "/" + self.c_channel + "/" + file
                    path_array.append(jpg)

        if shuffle_list is None:
            print('Reading~')

            shuffle_list = list(range(0, len(path_array)))

            for i in range(len(shuffle_list)):
                s_path_array.append(path_array[shuffle_list[i]])
        else:
            print('Reading~')

            for i in range(len(shuffle_list)):
                s_path_array.append(path_array[shuffle_list[i]])

        if self.c_channel is 'x':
            return s_path_array, shuffle_list
        else:
            return s_path_array

    def read_images_by_batch(self, batch_arr, batch_size):
        data_list = []

        for i in range(len(batch_arr)):
            img = Image.open(batch_arr[i])
            img.thumbnail((self.height, self.width), Image.AFFINE)
            arr = np.array(img).reshape((self.height, self.width, self.channel))
            data_list.append(arr)
            img.close()

        X1 = np.array(data_list)
        data_X = np.zeros((batch_size, self.height, self.width, self.channel), dtype=np.float32)

        for i in range(batch_size):
            data_X[i, :, :, :] = X1[i, :, :, :]

        return data_X / 255.
