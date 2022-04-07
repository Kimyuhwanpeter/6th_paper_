# -*- coding:utf-8 -*-
from random import shuffle, random
from tree_model import *
from tensorflow.keras import backend as K
from Cal_measurement import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/train.txt",

                           "test_txt_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/test.txt",
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerLabels_temp/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/apple_pear/FlowerImages/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/226/226",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "sample_images": "C:/Users/Yuhwan/Downloads/tt",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/checkpoint",

                           "save_print": "C:/Users/Yuhwan/Downloads/_.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",

                           "test_images": "C:/Users/Yuhwan/Downloads/test_images",

                           "train": True})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
optim2 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
color_map = np.array([[0, 0, 0],[255,0,0]], np.uint8)
def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.) 
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)
        
    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def false_dice_loss(y_true, y_pred):
    y_true = 1 - tf.cast(y_true, tf.float32)
    y_pred = 1 - tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean((alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               + ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0)))
        # return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
        #        -tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed

def two_region_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2*(tf.reduce_sum(y_true*y_pred) + tf.reduce_sum((1 - y_true)*(1 - y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred) + tf.reduce_sum(2 - y_true - y_pred)

    return 1 - tf.math.divide(numerator, denominator)

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)


def cal_loss(model, model2, images, labels, object_buf):
    
    with tf.GradientTape() as tape: # consider object and non-object
        batch_labels = tf.reshape(labels, [-1,])
        raw_logits = run_model(model, images, True)
        logits = tf.reshape(raw_logits, [-1, ])

        dice_loss = true_dice_loss(batch_labels, logits)

    grads = tape.gradient(dice_loss, model.trainable_variables)
    optim2.apply_gradients(zip(grads, model.trainable_variables))

    with tf.GradientTape() as tape2: # consider only object

        batch_labels = tf.reshape(labels, [-1,])
        raw_logits = run_model(model2, images * tf.nn.sigmoid(raw_logits), True)
        logits = tf.reshape(raw_logits, [-1, ])
        only_back_output = tf.where(batch_labels == 0, 1 - tf.nn.sigmoid(logits), logits)
        only_back_indices = tf.squeeze(tf.where(batch_labels == 0), -1)
        only_back_logits = tf.gather(only_back_output, only_back_indices)
        only_back_loss = tf.reduce_mean(-tf.math.log(only_back_logits + tf.keras.backend.epsilon()))

        only_object_output = tf.where(batch_labels == 1, tf.nn.sigmoid(logits), logits)
        only_object_indices = tf.squeeze(tf.where(batch_labels == 1), -1)
        only_object_logits = tf.gather(only_object_output, only_object_indices)
        only_object_loss = tf.reduce_mean(-tf.math.log(only_object_logits + tf.keras.backend.epsilon()))

        only_object_labels = tf.gather(batch_labels, only_object_indices)
        non_object_labels = tf.gather(batch_labels, only_back_indices)

        if object_buf[0] < object_buf[1]:
            distri_loss1 = binary_focal_loss(alpha=object_buf[1])(batch_labels, tf.nn.sigmoid(logits)) \
                + tf.keras.losses.BinaryCrossentropy(from_logits=True)(only_object_labels,
                                                                       only_object_logits)
            dice_loss = object_buf[1] * true_dice_loss(only_object_labels, 
                                                       only_object_logits) \
                + object_buf[0] * false_dice_loss(non_object_labels,
                                                  only_back_logits)
        else:
            distri_loss1 = binary_focal_loss(alpha=object_buf[0])(batch_labels, tf.nn.sigmoid(logits)) \
                + tf.keras.losses.BinaryCrossentropy(from_logits=True)(non_object_labels,
                                                                       only_back_logits)
            dice_loss = object_buf[0] * true_dice_loss(only_object_labels, 
                                                       only_object_logits) \
                + object_buf[1] * false_dice_loss(non_object_labels,
                                                  only_back_logits)

        total_loss = only_back_loss + only_object_loss + dice_loss + distri_loss1

    grads2 = tape2.gradient(total_loss, model2.trainable_variables)
    optim.apply_gradients(zip(grads2, model2.trainable_variables))

    return total_loss

def main():

    model = modified_network(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), nclasses=1)
    model2 = modified_network(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), nclasses=1)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, model2=model2, optim=optim, optim2=optim2)
        ckpt_manger = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manger.latest_checkpoint:
            ckpt.restore(ckpt_manger.latest_checkpoint)
            print("Restored!!!!!")

    if FLAGS.train:
        count = 0;

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)

                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 1, 0)

                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf += count_c_i_lab

                object_buf = class_imbal_labels_buf
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                object_buf = tf.nn.softmax(object_buf).numpy()

                loss = cal_loss(model, model2, batch_images, batch_labels, object_buf)

                if count % 10 == 0:
                    print("Epochs: {}, Loss = {} [{}/{}]".format(epoch, loss, step + 1, tr_idx))


                if count % 100 == 0:

                    object = run_model(model, batch_images, False)
                    object = tf.nn.sigmoid(object)
                    raw_logits = run_model(model2, batch_images * object, False)
                    object_output = tf.nn.sigmoid(raw_logits)
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i, :, :, 0], tf.int32).numpy()
                        object_image = object_output[i, :, :, 0]
                        object_image = tf.where(object_image >= 0.5, 1, 0).numpy()

                        pred_mask_color = color_map[object_image]
                        label_mask_color = color_map[label]
 
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)


                count += 1

            tr_iter = iter(train_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    object = run_model(model, batch_image, False)
                    object = tf.nn.sigmoid(object)
                    raw_logits = run_model(model2, batch_image * object, False)
                    object_output = tf.nn.sigmoid(raw_logits[0, :, :, 0])
                    object_output = tf.where(object_output >= 0.5, 1, 0).numpy()

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 255, 1, 0)

                    cm_ = Measurement(predict=object_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm += cm_

                    iou += cm_[1,1]/(cm_[1,1] + cm_[0,1] + cm_[1,0])
                    recall_ += cm_[1,1] / (cm_[1,1] + cm_[0,1])
                    precision_ += cm_[1,1] / (cm_[1,1] + cm_[1,0])

                precision_ = precision_ / len(test_img_dataset)
                recall_ = recall_ / len(test_img_dataset)
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("train mIoU = %.4f, train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (iou / len(train_img_dataset),
                                                                                                                        f1_score_,
                                                                                                                        recall_ / len(train_img_dataset),
                                                                                                                        precision_ / len(train_img_dataset)))

            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train IoU: ")
            output_text.write("%.4f" % (iou / len(train_img_dataset)))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_ / len(train_img_dataset)))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_ / len(train_img_dataset)))
            output_text.write("\n")

            test_iter = iter(test_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    object = run_model(model, batch_image, False)
                    object = tf.nn.sigmoid(object)
                    raw_logits = run_model(model2, batch_image * object, False)
                    object_output = tf.nn.sigmoid(raw_logits[0, :, :, 0])
                    object_output = tf.where(object_output >= 0.5, 1, 0).numpy()

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 255, 1, 0)

                    cm_ = Measurement(predict=object_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm += cm_

                    pred_mask_color = color_map[object_output]
                    label_mask_color = color_map[batch_label]

                    iou += cm_[1,1]/(cm_[1,1] + cm_[0,1] + cm_[1,0])
                    recall_ += cm_[1,1] / (cm_[1,1] + cm_[0,1])
                    precision_ += cm_[1,1] / (cm_[1,1] + cm_[1,0])

                precision_ = precision_ / len(test_img_dataset)
                recall_ = recall_ / len(test_img_dataset)
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)

                #name = test_img_dataset[i].split("/")[-1].split(".")[0]
                #plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                #plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)


            print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (iou / len(test_img_dataset),
                                                                                                                    f1_score_,
                                                                                                                    recall_ / len(test_img_dataset),
                                                                                                                    precision_ / len(test_img_dataset)))
            output_text.write("test IoU: ")
            output_text.write("%.4f" % (iou / len(test_img_dataset)))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_ / len(test_img_dataset)))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_ / len(test_img_dataset)))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            #model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            #if not os.path.isdir(model_dir):
            #    print("Make {} folder to store the weight!".format(epoch))
            #    os.makedirs(model_dir)
            #ckpt = tf.train.Checkpoint(model=model, model2=model2, optim=optim, optim2=optim2)
            #ckpt_dir = model_dir + "/apple_model_{}.ckpt".format(epoch)
            #ckpt.save(ckpt_dir)

if __name__ == "__main__":
    main()
