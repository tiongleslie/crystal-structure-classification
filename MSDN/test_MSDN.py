# ---------------------------------------------------------
# Example Code for Multi-stream DenseNet (MSDN) Implementation
# Licensed under The KIST License
# Written by CSRC, KIST
# ---------------------------------------------------------
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import sys
import tensorflow.python.util.deprecation as deprecation
from PIL import Image
from MSDN_utils import MSDN_utils
from image_utils import image_utils


# Hide all the warning messages from TensorFlow
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', help='Test dataset path.', default='Test Sample')
    parser.add_argument('--model_dir', help='Pretrained model path.', default='pretrained model')
    parser.add_argument('--batch_size', help='Batch size.', default=32)
    parser.add_argument('--plot_sample', help='Demonstrate the results.', default=True)

    return parser.parse_args(argv)


def plot_sample(valid_data_r, valid_data_g, valid_data_b, i, c_s, s_g):
    mpl.rcParams['toolbar'] = 'None'
    f = plt.figure(i ,figsize=(4, 3))
    f.patch.set_facecolor('black')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in f.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    string = "Sample " + str(i) + "\nCrystal System: " + str(c_s) + "\nSpace Group: " + str(s_g) + "\n\n"
    f.add_subplot(1, 3, 1)
    plt.text(10, 0, string, fontsize=10, color='white')
    img_r = Image.open(valid_data_r)
    plt.imshow(img_r)
    plt.gca().set_axis_off()
    img_r.close()
    f.add_subplot(1, 3, 2)
    img_g = Image.open(valid_data_g)
    plt.imshow(img_g)
    plt.gca().set_axis_off()
    img_g.close()
    f.add_subplot(1, 3, 3)
    img_b = Image.open(valid_data_b)
    plt.imshow(img_b)
    img_b.close()
    plt.gca().set_axis_off()
    plt.show(block=True)


def main(args):
    batch_size = args.batch_size
    total_score = []
    test_path = args.test_dir

    #
    # Test Data
    #
    print("======================================")
    print("=============Data Loading=============")
    print("======================================")

    # Read the images from R
    vim_r = image_utils(path=test_path, c_channel='x')
    valid_data_r, vshuffle_list = vim_r.load_data()
    print('R images (test_data) are loaded')

    # Read the images from G
    vim_g = image_utils(path=test_path, c_channel='y')
    valid_data_g = vim_g.load_data(shuffle_list=vshuffle_list)
    print('G images (test_data) are loaded')

    # Read the images from B
    vim_b = image_utils(path=test_path, c_channel='z')
    valid_data_b = vim_b.load_data(shuffle_list=vshuffle_list)
    print('B images (test_data) are loaded')

    batch_arr_valid = range(0, len(valid_data_r), batch_size)
    max_valid = batch_arr_valid[len(batch_arr_valid) - 1]

    #
    # Load Space Groups' Label Info
    #
    SG = np.asarray(sio.loadmat('data info/SG.mat')['SG'])

    #
    # Load the data by batch
    #
    print("\n\n======================================")
    print("=====Load Pretrained Model: MSDN======")
    print("======================================")
    for k in range(len(batch_arr_valid)):
        if batch_arr_valid[k] == max_valid:
            test_range_r = valid_data_r[batch_arr_valid[k]:len(valid_data_r)]
            test_range_g = valid_data_g[batch_arr_valid[k]:len(valid_data_g)]
            test_range_b = valid_data_b[batch_arr_valid[k]:len(valid_data_b)]
            test_image_batch_r = vim_r.read_images_by_batch(test_range_r, len(test_range_r))
            test_image_batch_g = vim_g.read_images_by_batch(test_range_g, len(test_range_g))
            test_image_batch_b = vim_b.read_images_by_batch(test_range_b, len(test_range_b))
        else:
            test_range_r = valid_data_r[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            test_range_g = valid_data_g[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            test_range_b = valid_data_b[batch_arr_valid[k]:batch_arr_valid[k] + batch_size]
            test_image_batch_r = vim_r.read_images_by_batch(test_range_r, batch_size)
            test_image_batch_g = vim_g.read_images_by_batch(test_range_g, batch_size)
            test_image_batch_b = vim_b.read_images_by_batch(test_range_b, batch_size)

        # call the test function
        score = MSDN_utils.test_MSDN(path=args.model_dir, test_image_r=test_image_batch_r,
                                     test_image_g=test_image_batch_g,
                                     test_image_b=test_image_batch_b)
        total_score.extend(score)

    sample = np.array(total_score)
    print("Completed test!")
    sio.savemat('test_scores.mat', {'scores': sample})
    print("\n\n======================================")
    print("========Classification Result=========")
    print("======================================")
    print("Number of samples tested: %i\n" % (len(sample)))
    for i in range(len(sample)):
        s_g = SG[0][np.argmax(sample[i])]
        c_s = MSDN_utils.disp_crystal_struc(SG[0][np.argmax(sample[i])])
        print("Sample %i\nCrystal System: %s\nSpace Group: %i\n\n" % (i + 1, c_s, s_g))
        if args.plot_sample is True:
            plot_sample(valid_data_r[i], valid_data_g[i], valid_data_b[i], i+1, c_s, s_g)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
