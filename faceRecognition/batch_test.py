# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : batch_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:33:30
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils import file_processing,image_processing
import face_recognition
import predict
resize_width = 160
resize_height = 160

def face_recognition_for_bzl(model_path,test_dataset, filename):
    # 加载数据库的数据
    dataset_emb,names_list=predict.load_dataset(dataset_path, filename)
    print("loadind dataset...\n names_list:{}".format(names_list))
    # 初始化mtcnn人脸检测
    face_detect=face_recognition.Facedetection()
    # 初始化facenet
    face_net=face_recognition.facenetEmbedding(model_path)

    #获得测试图片的路径和label
    filePath_list, label_list=file_processing.gen_files_labels(test_dataset)
    label_list=[name.split('_')[0] for name in label_list]
    print("filePath_list:{},label_list{}".format(len(filePath_list),len(label_list)))

    right_num = 0
    wrong_num = 0
    detection_num = 0
    test_num = len(filePath_list)
    for image_path, label_name in zip(filePath_list, label_list):
        print("image_path:{}".format(image_path))
        # 读取图片
        image = image_processing.read_image_gbk(image_path)
        # 人脸检测
        bboxes, landmarks = face_detect.detect_face(image)
        bboxes, landmarks =face_detect.get_square_bboxes(bboxes, landmarks,fixed="height")
        if bboxes == [] or landmarks == []:
            print("-----no face")
            continue
        if len(bboxes) >= 2 or len(landmarks) >= 2:
            print("-----image have {} faces".format(len(bboxes)))
            continue
        # 获得人脸框区域
        face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width)
        face_images = image_processing.get_prewhiten_images(face_images,normalization=True)
        # face_images = image_processing.get_prewhiten_images(face_images,normalization=True)

        pred_emb=face_net.get_embedding(face_images)
        pred_name,pred_score=predict.compare_embadding(pred_emb, dataset_emb, names_list,threshold=1.3)
        # 在图像上绘制人脸边框和识别的结果
        # show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]
        # image_processing.show_image_text("face_recognition", image, bboxes, show_info)

        index=0
        pred_name  = pred_name[index]
        pred_score = pred_score[index]
        if pred_name == label_name:
            right_num += 1
        else:
            wrong_num += 1
        detection_num += 1
        print("-------------label_name:{},pred_name:{},score:{:3.4f},status:{}".format(label_name, pred_name, pred_score,
                                                                                       (label_name == pred_name)))
    # 准确率
    accuracy = right_num / detection_num
    # 漏检率
    misdetection = (test_num - detection_num) / test_num
    print("-------------right_num/detection_num:{}/{},accuracy rate:{}".format(right_num, detection_num, accuracy))
    print("-------------misdetection/all_num:{}/{},misdetection rate:{}".format((test_num - detection_num), test_num,
                                                                                misdetection))

if __name__=='__main__':
    model_path='models/20180408-102900'
    dataset_path='dataset/emb/faceEmbedding.npy'
    filename='dataset/emb/name.txt'
    # image_path='dataset/test_images/1.jpg'
    # test_dataset='E:/Face/dataset/bzl/test2/test_dataset'
    # test_dataset='E:/Face/dataset/bzl/test3/test_dataset'
    test_dataset="E:/Face/dataset/bzl/clear_data2"

    face_recognition_for_bzl(model_path, test_dataset, filename)