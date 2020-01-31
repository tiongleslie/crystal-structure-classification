# ---------------------------------------------------------
# Multi-stream DenseNet (MSDN) Implementation
# Licensed under The KIST License
# Written by CSRC, KIST
# ---------------------------------------------------------
import tensorflow as tf
import os
import tensorflow.python.util.deprecation as deprecation

# Hide all the warning messages from TensorFlow
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class MSDN_utils(object):
    def __init__(self):
        print('Model Loading...')

    def test_MSDN(path, test_image_r, test_image_g, test_image_b):
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '/MSDN.meta')
            saver.restore(sess, path + '/MSDN')

            x1_tfph = graph.get_tensor_by_name("input_img_x:0")
            x2_tfph = graph.get_tensor_by_name("input_img_y:0")
            x3_tfph = graph.get_tensor_by_name("input_img_z:0")
            drop_prob = graph.get_tensor_by_name("keep_prob:0")

            pred1 = graph.get_tensor_by_name("output_1/add:0")
            pred2 = graph.get_tensor_by_name("output_2/add:0")
            pred3 = graph.get_tensor_by_name("output_3/add:0")

            output_score = tf.nn.softmax(pred1 + pred2 + pred3, name="softmax_scores_1")

            test_feed = {x1_tfph: test_image_r, x2_tfph: test_image_g, x3_tfph: test_image_b, drop_prob: 1.0}
            score = sess.run(output_score, feed_dict=test_feed)

        return score

    def disp_crystal_struc(SG):
        if 1 <= SG <= 2:
            return "Triclinic"
        elif 3 <= SG <= 15:
            return "Monoclinic"
        elif 16 <= SG <= 74:
            return "Orthorhombic"
        elif 75 <= SG <= 142:
            return "Tetragonal"
        elif 143 <= SG <= 167:
            return "Trigonal"
        elif 168 <= SG <= 194:
            return "Hexagonal"
        elif 195 <= SG <= 230:
            return "Cubic"
