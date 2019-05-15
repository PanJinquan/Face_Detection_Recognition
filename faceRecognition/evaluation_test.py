# -*- coding: utf-8 -*-

import os
import pickle
from utils import image_processing,file_processing,evaluation
import cv2
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import face_recognition

def load_npy(dir_path):
    issames_path=os.path.join(dir_path,"issames.npy")
    pred_score_path=os.path.join(dir_path,"pred_score.npy")
    issames=np.load(issames_path)
    pred_score=np.load(pred_score_path)
    return pred_score,issames

def load_bin(path, image_size=[112,112]):
    '''
    加载人脸bin文件数据，bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    :param path:
    :param image_size:
    :return:
    '''
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data=[]
    for i in range(len(bins)):
        _bin = bins[i]
        img = cv2.imdecode(_bin, -1) # opencv image is bgr
        img = image_processing.resize_image(img,resize_height=image_size[0],resize_width=image_size[1])
        # image_processing.show_image("src",img)
        data.append(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    data = np.array(data)
    issames = np.array(issame_list)
    return data, issames

def split_data(data):
    '''
    按照奇偶项分割数据
    :param data:
    :return:
    '''
    data1 = data[0::2]
    data2 = data[1::2]
    return data1,data2

def get_pair_scores(faces_data, issames_data, model_path, save_path=None):
    '''
    计算分数
    :param faces_data:
    :param issames_data:
    :param model_path: insightFace模型路径
    :param save_path:
    :return:
    '''
    faces_list1,faces_list2 = split_data(faces_data)
    face_net=face_recognition.facenetEmbedding(model_path)

    pred_score=[]
    i=0
    for face1,face2,issame in zip(faces_list1, faces_list2, issames_data):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        face_images1 = image_processing.get_prewhiten_images([face1],normalization=False)
        face_images2 = image_processing.get_prewhiten_images([face2],normalization=False)

        face_emb1=face_net.get_embedding(face_images1)
        face_emb2=face_net.get_embedding(face_images2)

        # face_emb1 = face_net.get_faces_embedding([face1])
        # face_emb2 = face_net.get_faces_embedding([face2])
        dist = np.sqrt(np.sum(np.square(np.subtract(face_emb1, face_emb2))))
        pred_score.append(dist)
        i += 1
        if i % 100 == 0:
            print('processing data :', i)
    pred_score=np.array(pred_score).reshape(-1)
    issames_data= issames_data + 0 # 将true和false转为1/0
    if save_path is not None:
        issames_path = os.path.join(save_path, "issames.npy")
        pred_score_path = os.path.join(save_path, "pred_score.npy")
        np.save(issames_path, issames_data)
        np.save(pred_score_path,pred_score)
    return pred_score, issames_data

if __name__=='__main__':
    # bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    bin_path='./dataset/faces_emore/agedb_30.bin'
    model_path = './models/20180408-102900'
    # 加载bin测试数据
    faces_data, issames_data=load_bin(bin_path,image_size=[160,160])

    # 计算分数
    save_path="./dataset/faces_emore"
    pred_score, issames_data=get_pair_scores(faces_data, issames_data, model_path, save_path=save_path)
    pred_score, issames_data=load_npy(dir_path=save_path)

    # 计算roc曲线
    fpr, tpr, roc_auc, threshold, optimal_idx=evaluation.get_roc_curve(y_true=issames_data, y_score=pred_score, invert=True, plot_roc=True)

    print("fpr:{}".format(fpr))
    print("tpr:{}".format(tpr))
    print("threshold:{}".format(threshold))
    print("roc_auc:{}".format(roc_auc))
    print("optimal_idx :{},best_threshold :{} ".format(optimal_idx,threshold[optimal_idx]))

    # 测评数据保存
    evaluation.save_evaluation(fpr, tpr, roc_auc, "evaluation.npz")

    # 加载测评数据
    fpr, tpr, roc_auc=evaluation.load_evaluation("evaluation.npz")
    evaluation.plot_roc_curve(fpr_list=[fpr], tpr_list=[tpr], roc_auc_list=[roc_auc], line_names=["FaceNet"])
