# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : file_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 10:11:56
"""
import os
# import pandas as pd

def write_data(file, content_list, model):
    with open(file, mode=model) as f:
        for line in content_list:
            f.write(line + "\n")


def read_data(file):
    with open(file, mode="r") as f:
        content_list = f.readlines()
        content_list = [content.rstrip() for content in content_list]
    return content_list

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


if __name__ == '__main__':
    file_dir = 'JPEGImages'
    file_list = get_files_list(file_dir)
    for file in file_list:
        print(file)
