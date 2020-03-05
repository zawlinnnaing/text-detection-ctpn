# coding=utf-8

import tensorflow as tf
import numpy as np
import cv2
import time
import shutil
import os
from PIL import Image
import sys
import app_config as cfg

import add_sys_path  # Place this import before import modules from project.
from nets import model_train as model
from utils.image_processing import transform_img
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector


tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def name_from_path(file_path: str):
    return os.path.basename(os.path.splitext(file_path)[0])


# def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def crop_bbox(bbox: list, img):
    """
    bbox -> bounding box coordinates
    img -> (np.array) Image to be cropped

    return: img - np.array
    """
    x_min_index = 0  # Indexes for getting values in bounding box array.
    # Refer to this file /utils/text_connector/text_proposal_connector.py for how to get these indexes.
    y_min_index = 1
    x_max_index = 2
    y_max_index = 5
    (x_min, x_max, y_min, y_max) = (
        bbox[x_min_index], bbox[x_max_index], bbox[y_min_index], bbox[y_max_index])
    width = x_max-x_min
    height = y_max-y_min
    new_img = np.array(img, copy=True)
    return new_img[y_min: y_max, x_min: x_max], x_min, x_max, y_min, y_max


def text_extraction(image, image_name, x, y, x2, y2, output_dir):
    (w, h) = (x2-x, y2-y)
    if h < 20 or w < 35:
        print("Image is too small. Discarded image {}".format(image_name))
        return
    size = (1024, 28)

    # _, im_bw = cv2.threshold(
    #     gray, 110, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # im_bw = cv2.dilate(im_bw, kernel, iterations=1)  # dilate
    # im_bw = cv2.erode(im_bw, kernel, iterations=1)  # erosion
    im_bw = transform_img(image)
    img = cv2.resize(im_bw, (1024, 28))
    img = img.astype(np.float32)
    img /= 255

    (x2, y2) = (x+w, y+h)
    fname = "{0}_{1}x{2}_{3}x{4}.jpg".format(image_name, x, y, x2, y2)
    file_name = os.path.join(output_dir, fname)
    cv2.imwrite(file_name, im_bw)

    im = Image.open(file_name)
    im.thumbnail(size, Image.ANTIALIAS)
    new_im = Image.new("RGB", size, "black")  # luckily, this is already black!
    new_im.paste(im, (0, 0))
    new_im.save(file_name, format='JPEG', subsampling=0, quality=100)
    return (new_im)


def predict(im_fn):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CHECKPOINT_PATH

    with tf.get_default_graph().as_default():
        # Building Tensorflow Graph
        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Computing TF graph
            ckpt_state = tf.train.get_checkpoint_state(cfg.CHECKPOINT_PATH)
            model_path = os.path.join(cfg.CHECKPOINT_PATH, os.path.basename(
                ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            print("Making directory for storing cropped images ... ")
            cropped_dir = os.path.join(
                cfg.OUTPUT_DIR, name_from_path(im_fn))
            os.makedirs(cropped_dir, exist_ok=True)

            print('===============')
            print(im_fn)
            start = time.time()
            try:
                im = cv2.imread(im_fn)[:, :, ::-1]
            except:
                print("Error reading image {}!".format(im_fn))
                return

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})
        textsegs, _ = proposal_layer(
            cls_prob_val, bbox_pred_val, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]

        textdetector = TextDetector(DETECT_MODE='H')
        boxes = textdetector.detect(
            textsegs, scores[:, np.newaxis], img.shape[:2])

        boxes = np.array(boxes, dtype=np.int)

        cost_time = (time.time() - start)
        print("cost time: {:.2f}s".format(cost_time))

        for i, box in enumerate(boxes):
            # Cropped image in bounding box and save in specified directory
            cropped_img, x_min, x_max, y_min, y_max = crop_bbox(
                box[:8], img)
            text_extraction(
                cropped_img, name_from_path(im_fn), x_min, y_min, x_max, y_max, cropped_dir)

            # with open(os.path.join(cfg.OUTPUT_DIR, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
            #           "w") as f:
            #     for i, box in enumerate(boxes):
            #         line = ",".join(str(box[k]) for k in range(8))
            #         line += "," + str(scores[i]) + "\r\n"
            #         f.writelines(line)
            # break

        return cropped_dir
