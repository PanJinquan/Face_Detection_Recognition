# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : create_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:31:09
"""
import numpy as np
from utils import image_processing , file_processing,debug
import face_recognition
import cv2
import os

resize_width = 160
resize_height = 160


def get_face_embedding(model_path,files_list, names_list):
    '''
    获得embedding数据
    :param files_list: 图像列表
    :param names_list: 与files_list一一的名称列表
    :return:
    '''
    # 转换颜色空间RGB or BGR
    colorSpace="RGB"
    # 初始化mtcnn人脸检测
    face_detect = face_recognition.Facedetection()
    # 初始化facenet
    face_net = face_recognition.facenetEmbedding(model_path)

    embeddings=[] # 用于保存人脸特征数据库
    label_list=[] # 保存人脸label的名称，与embeddings一一对应
    for image_path, name in zip(files_list, names_list):
        print("processing image :{}".format(image_path))
        # image_path='E:/Face/dataset/bzl/subjectphoto_with_name/谢伟林_179_180.jpg'
        image = image_processing.read_image_gbk(image_path, colorSpace=colorSpace)
        # 进行人脸检测，获得bounding_box
        bboxes, landmarks = face_detect.detect_face(image)
        bboxes, landmarks =face_detect.get_square_bboxes(bboxes, landmarks,fixed="height")
        # image_processing.show_image_boxes("image",image,bboxes)
        if bboxes == [] or landmarks == []:
            print("-----no face")
            continue
        if len(bboxes) >= 2 or len(landmarks) >= 2:
            print("-----image have {} faces".format(len(bboxes)))
            continue
        # 获得人脸区域
        face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width)
        # 人脸预处理，归一化
        face_images = image_processing.get_prewhiten_images(face_images,normalization=True)
        # 获得人脸特征
        pred_emb = face_net.get_embedding(face_images)
        embeddings.append(pred_emb)
        # 可以选择保存image_list或者names_list作为人脸的标签
        # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
        label_list.append(image_path)
        # label_list.append(name)
    return embeddings,label_list

def create_face_embedding(model_path,dataset_path,out_emb_path,out_filename):
    '''

    :param model_path: faceNet模型路径
    :param dataset_path: 人脸数据库路径，每一类单独一个文件夹
    :param out_emb_path: 输出embeddings的路径
    :param out_filename: 输出与embeddings一一对应的标签
    :return: None
    '''
    files_list,names_list=file_processing.gen_files_labels(dataset_path,postfix='jpg')
    embeddings,label_list=get_face_embedding(model_path,files_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))

    embeddings=np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_data(out_filename, label_list, model='w')

def create_face_embedding_for_bzl(model_path,dataset_path,out_emb_path,out_filename):
    '''
    :param model_path: faceNet模型路径
    :param dataset_path: 人脸数据库路径，图片命名方式：张三_XXX_XXX.jpg,其中“张三”即为label
    :param out_emb_path: 输出embeddings的路径
    :param out_filename: 输出与embeddings一一对应的标签
    :return: None
    '''
    image_list = file_processing.get_images_list(dataset_path, postfix=['*.jpg', '*.png'])
    names_list=[]
    for image_path in image_list:
        basename = os.path.basename(image_path)
        names = basename.split('_')[0]
        names_list.append(names)
    embeddings,label_list=get_face_embedding(model_path,image_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))
    embeddings=np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_data(out_filename, label_list, model='w')

if __name__ == '__main__':
    model_path = 'models/20180408-102900'
    dataset_path='dataset/images'
    out_emb_path = 'dataset/emb/faceEmbedding.npy'
    out_filename = 'dataset/emb/name.txt'
    create_face_embedding(model_path, dataset_path,out_emb_path, out_filename)
    # dataset_path = 'E:/Face/dataset/bzl/test2/facebank'
    # dataset_path = 'E:/Face/dataset/bzl/test3/facebank'
    # dataset_path='E:/Face/dataset/bzl/subjectphoto_with_name' #人脸数据库的路径
    # create_face_embedding_for_bzl(model_path, dataset_path,out_emb_path, out_filename)