#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from skimage import transform
import tensorflow as tf
import numpy as np
import cv2


# the function of candidate center detection, which uses CNN

def detection(img_original, stride):
    patch_size = 71
    height = np.size(img_original, 0)
    width = np.size(img_original, 1)
    img_original = transform.resize(img_original, (height, width))
    imgs = []
    coordidate = []
    for i in range(patch_size, height-patch_size, stride):
        for j in range(patch_size, width-patch_size, stride):
            img_original_patch = img_original[int(i-(patch_size-1)/2):int(
                i+(patch_size-1)/2+1), int(j-(patch_size-1)/2):int(j+(patch_size-1)/2+1), :]
            imgs.append(img_original_patch)
            coordidate.append([i, j])
    data = np.asarray(imgs, np.float32)
    output = []
    with  tf.compat.v1.Session() as sess:
       
        
        saver =  tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(sess,  tf.compat.v1.train.latest_checkpoint('model/'))
        graph = tf.compat.v1.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        vol_slice = 5000
        num_slice = math.ceil(np.size(data, 0)/vol_slice)
        for i in range(0, num_slice, 1):
            if i+1 != num_slice:
                data_temp = data[i*vol_slice:(i+1)*vol_slice]
            else:
                data_temp = data[i*vol_slice:np.size(data, 0)]

            feed_dict = {x: data_temp}
            logits = graph.get_tensor_by_name("logits_eval:0")
            classification_result = sess.run(logits, feed_dict)
            output_temp = tf.argmax(classification_result, 1).eval()
            output = np.hstack((output, output_temp))
    candidate_center = []
    for i in range(len(output)):
        if output[i] == 1:
            candidate_center.append(coordidate[i])
    return np.array(candidate_center)  # the matrix of candidate center

# the function of clustering, which gets the group of candidate center


def clustering(candidate_center, threshold_dis):
    x = candidate_center[:, 1]
    y = candidate_center[:, 0]
    group_distance = []
    for i in range(len(candidate_center)):
        xpoint, ypoint = x[i], y[i]
        xTemp, yTemp = x, y
        distance = np.sqrt(pow((xpoint-xTemp), 2)+pow((ypoint-yTemp), 2))
        distance_matrix = np.vstack(
            (np.array(range(len(candidate_center))), distance))
        distance_matrix = np.transpose(distance_matrix)
        distance_sort = distance_matrix[distance_matrix[:, 1].argsort()]
        distance_sort = np.delete(distance_sort, 0, axis=0)
        thre_matrix = distance_sort[distance_sort[:, 1] <= threshold_dis]
        thre_point = thre_matrix[:, 0]
        thre_point = thre_point.astype(np.int64)
        thre_point = thre_point.tolist()
        thre_point.insert(0, i)
        group_distance.append(thre_point)

    group_clustering = [[]]

    for i in range(len(candidate_center)):
        m1 = group_distance[i]
        for j in range(len(group_clustering)):
            m2 = group_clustering[j]
            com = set(m2).intersection(set(m1))
            if len(com) == 0:
                if j == len(group_clustering)-1:
                    group_clustering.append(m1)
            else:
                m = set(m1).union(set(m2))
                group_clustering[j] = []
                group_clustering[j] = list(m)
                break
    group_clustering.pop(0)
    return group_clustering  # the group of candiate center

# the function of clustering the final center


def center_clustering(candidate_center, group_clustering):
    final_result = []
    for i in range(len(group_clustering)):
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:, 1]
        yz = points_coord[:, 0]
        x_mean = np.mean(xz)
        y_mean = np.mean(yz)
        final_result.append([y_mean, x_mean])
    final_result = np.array(final_result)
    final_result = final_result.astype(np.int64)
    return final_result  # the matrix of final center of steel bars

# these functions are showing the result which  include the :
# result of candidate center,


def show_original_red_point(img_original, candidate_center):
    cv2.namedWindow('original red point')
    for i in range(len(candidate_center)):
        cv2.circle(
            img_original, (candidate_center[i, 1], candidate_center[i, 0]), 2, (0, 0, 255), -1)
    cv2.putText(img_original, "result of candidate center",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 224, 60), 2)
    cv2.imshow('original red point', img_original)

# the bounding-box of clustering,


def show_green_box(img_original, candidate_center, group_clustering):
    cv2.namedWindow('green box')
    for i in range(len(candidate_center)):
        cv2.circle(
            img_original, (candidate_center[i, 1], candidate_center[i, 0]), 2, (0, 0, 255), -1)
    for i in range(len(group_clustering)):
        points_coord = candidate_center[group_clustering[i]]
        xz = points_coord[:, 1]
        yz = points_coord[:, 0]
        cv2.rectangle(img_original, (min(xz)-5, min(yz)-5),
                      (max(xz)+5, max(yz)+5), (0, 255, 0))
    cv2.putText(img_original, "bounding-box of clustering, ",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 224, 60), 2)
    cv2.imshow('green box', img_original)

# the center of clustering


def show_clustering_red_point(img_original, center_cluster):
    cv2.namedWindow('clustering red point')
    for i in range(len(center_cluster)):
        cv2.circle(
            img_original, (center_cluster[i, 1], center_cluster[i, 0]), 2, (0, 0, 255), -1)

    cv2.putText(img_original, " Final Steel Count: " + str(len(center_cluster)),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 224, 60), 2)
    cv2.imshow('clustering red point', img_original)


if __name__ == "__main__":
   
    img_original = cv2.imread('image/test.png')
    stride = 6 # the parameter of slide window stride
    candidate_center = detection(img_original, stride)

    distance_threshold = 20  # the parameter of distance clustering threshold
    group_clustering = clustering(candidate_center, distance_threshold)
  
    center_cluster = center_clustering(candidate_center, group_clustering)
    
    # show the result
    # show_original_red_point(img_original,candidate_center)
    # show_green_box(img_original,candidate_center,group_clustering)
    show_clustering_red_point(img_original, center_cluster)
    cv2.waitKey(0)
