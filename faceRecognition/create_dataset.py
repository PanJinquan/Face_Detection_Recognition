# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : create_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:31:09
"""
import numpy as np
from utils import image_processing , file_processing
import face_recognition
import cv2
import os

resize_width = 160
resize_height = 160

def create_face(images_dir, out_face_dir):
    '''
    生成人脸数据图库，保存在out_face_dir中，这些数据库将用于生成embedding数据库
    :param images_dir:
    :param out_face_dir:
    :return:
    '''
    # image_list=file_processing.get_files_list(images_dir, postfix='jpg')
    image_list,names_list=file_processing.gen_files_labels(images_dir,postfix='jpg')
    face_detect=face_recognition.Facedetection()
    for image_path ,name in zip(image_list,names_list):
        image=image_processing.read_image(image_path, resize_height=0, resize_width=0, normalization=False)
        # 获取 判断标识 bounding_box crop_image
        bounding_box, points = face_detect.detect_face(image)
        bounding_box = bounding_box[:,0:4].astype(int)
        bounding_box=bounding_box[0,:]
        print("face box:{}".format(bounding_box))
        face_image = image_processing.crop_image(image,bounding_box)
        # image_processing.show_image("face", face_image)
        # image_processing.show_image_box("face",image,bounding_box)
        out_path=os.path.join(out_face_dir,name)
        face_image=image_processing.resize_image(face_image, resize_height, resize_width)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        basename=os.path.basename(image_path)
        out_path=os.path.join(out_path,basename)
        image_processing.save_image(out_path,face_image)
        # cv2.waitKey(0)

def create_embedding(model_path, emb_face_dir, out_emb_path, out_filename):
    '''
    产生embedding数据库，保存在out_data_path中，这些embedding其实就是人脸的特征
    :param model_path:
    :param emb_face_dir:
    :param out_emb_path:
    :param out_filename:
    :return:
    '''
    face_net = face_recognition.facenetEmbedding(model_path)
    # image_list=file_processing.get_files_list(emb_face_dir,postfix='jpg')
    image_list,names_list=file_processing.gen_files_labels(emb_face_dir,postfix='jpg')
    images= image_processing.get_images(image_list,resize_height,resize_width,whiten=True)
    compare_emb = face_net.get_embedding(images)
    np.save(out_emb_path, compare_emb)

    # 可以选择保存image_list或者names_list作为人脸的标签
    # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
    file_processing.write_data(out_filename, image_list, model='w')

if __name__ == '__main__':
    images_dir='dataset/images'
    out_face_dir='dataset/emb_face'
    create_face(images_dir,out_face_dir)


    model_path = 'models/20180408-102900'
    emb_face_dir = './dataset/emb_face'
    out_emb_path = 'dataset/emb/faceEmbedding.npy'
    out_filename = 'dataset/emb/name.txt'
    create_embedding(model_path, emb_face_dir, out_emb_path, out_filename)