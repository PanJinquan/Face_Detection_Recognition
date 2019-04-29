# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : file_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:08:19
"""
import glob
import os
import os,shutil
import numpy as np

import pandas as pd

def write_data(file, content_list, model):
    with open(file, mode=model,encoding='utf-8') as f:
        for line in content_list:
            f.write(line + "\n")


def read_data(file,split=True):
    '''
    :param file:
    :return:
    '''
    with open(file, mode="r",encoding='utf-8') as f:
        content_list = f.readlines()
        if split:
            # 按空格分隔
            content_list = [content.rstrip().split(" ") for content in content_list]
        else:
            content_list = [content.rstrip() for content in content_list]
    return content_list

def get_images_list(image_dir,postfix=['*.jpg'],basename=False):
    '''
    获得文件列表
    :param image_dir: 图片文件目录
    :param postfix: 后缀名，可是多个如，['*.jpg','*.png']
    :param basename: 返回的列表是文件名（True），还是文件的完整路径(False)
    :return:
    '''
    images_list=[]
    for format in postfix:
        image_format=os.path.join(image_dir,format)
        image_list=glob.glob(image_format)
        if not image_list==[]:
            images_list+=image_list
    images_list=sorted(images_list)
    if basename:
        images_list=get_basename(images_list)
    return images_list

def get_basename(file_list):
    dest_list=[]
    for file_path in file_list:
        basename=os.path.basename(file_path)
        dest_list.append(basename)
    return dest_list

def copyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        # print("copy %s -> %s"%( srcfile,dstfile))


def merge_list(data1, data2):
    '''
    将两个list进行合并
    :param data1:
    :param data2:
    :return:返回合并后的list
    '''
    if not len(data1) == len(data2):
        return
    all_data = []
    for d1, d2 in zip(data1, data2):
        all_data.append(d1 + d2)
    return all_data


def split_list(data, split_index=1):
    '''
    将data切分成两部分
    :param data: list
    :param split_index: 切分的位置
    :return:
    '''
    data1 = []
    data2 = []
    for d in data:
        d1 = d[0:split_index]
        d2 = d[split_index:]
        data1+=d1
        data2+=d2
    return data1, data2


def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    '''
    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


def gen_files_labels(files_dir,postfix='ALL'):
    '''
    获取files_dir路径下所有文件路径，以及labels,其中labels用子级文件名表示
    files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
    :param files_dir:
    :postfix 后缀名
    :return:filePath_list所有文件的路径,label_list对应的labels
    '''
    # filePath_list = getFilePathList(files_dir)
    filePath_list=get_files_list(files_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))

    # 标签统计计数
    # print(pd.value_counts(label_list))
    return filePath_list, label_list

if __name__=='__main__':
    # image_dir='E:/git/dataset/tgs-salt-identification-challenge/train/images'
    image_dir='E:/git/dataset/tgs-salt-identification-challenge/train/my_images'

    image_list=get_images_list(image_dir, postfix=['*.png','*.jpg'],basename=True)
    train_file='my_train.txt'
    val_file='val.txt'

    train_list,val_list=split_list([image_list], split_index=int(1.0*len(image_list)))
    write_data(train_file, train_list, model='w')
    write_data(val_file, val_list, model='w')
    data=read_data(train_file)
    print("data")