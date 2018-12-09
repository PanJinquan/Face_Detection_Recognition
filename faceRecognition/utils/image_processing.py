# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : image_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 10:10:27
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

###############################################图片显示############################################
def show_image(win_name, rgb_image):
    plt.title(win_name)
    plt.imshow(rgb_image)
    plt.show()
def show_image_rect(win_name, rgb_image, rect):
    '''
    :param win_name:
    :param rgb_image:
    :param rect: x,y,w,h
    :return:
    '''
    plt.figure()
    plt.title(win_name)
    plt.imshow(rgb_image)
    rect =plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()

def show_image_boxes(win_name, rgb_image, boxes):
    '''
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画矩行
    :param rgb_image:
    :param bounding_boxes:[[x1,y1,x2,y2]]
    :return:
    '''
    plt.title(win_name)
    for box in boxes:
        # box = box.astype(int)
        cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    plt.imshow(rgb_image)
    plt.show()

def show_image_box(win_name, rgb_image, box):
    '''
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画矩行
    :param rgb_image:
    :param bounding_boxes:[[x1,y1,x2,y2]]
    :return:
    '''
    plt.title(win_name)
    # box = box.astype(int)
    cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    plt.imshow(rgb_image)
    plt.show()

def cv_show_image_text(win_name,bgr_image, boxes,boxes_name):
    '''

    :param boxes_name:
    :param bgr_image: bgr image
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    '''

    for name ,box in zip(boxes_name,boxes):
        cv2.rectangle(bgr_image, (box[0],box[1]),(box[2],box[3]), (0, 255, 0), 2, 8, 0)
        cv2.putText(bgr_image,name, (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), thickness=2)
    cv2.imshow(win_name, bgr_image)
    cv2.waitKey(30)



###############################################图片读取和保存############################################
def read_image(image_path, resize_height=0, resize_width=0, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param image_path:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", image_path)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image

def save_image(image_path, rgb_image):
    plt.imsave(image_path, rgb_image)

###############################################图片裁剪############################################
def crop_image(image, box):
    '''
    :param image: rgb image
    :param box: [x1,y1,x2,y2]
    :return:
    '''
    crop_img= image[box[1]:box[3], box[0]:box[2]]
    return crop_img

def crop_images(image, boxes, resize_height=0, resize_width=0):
    '''
    :param image: rgb image
    :param boxes:  [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :param resize_height:
    :param resize_width:
    :return:
    '''
    crops=[]
    for box in boxes:
        crop_img=crop_image(image, box)
        if resize_height > 0 and resize_width > 0:
            crop_img = cv2.resize(crop_img, (resize_width, resize_height))
        crops.append(crop_img)
    crops=np.stack(crops)
    return crops

def get_crop_images(image, boxes, resize_height=0, resize_width=0, whiten=False):
    '''

    :param image: rgb image
    :param boxes:[[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :param resize_height:
    :param resize_width:
    :param whiten:
    :return:
    '''
    crops=[]
    for box in boxes:
        crop_img=crop_image(image, box)
        if resize_height > 0 and resize_width > 0:
            crop_img = cv2.resize(crop_img, (resize_width, resize_height))
        if whiten:
            crop_img = prewhiten(crop_img)
        crops.append(crop_img)
    crops=np.stack(crops)
    return crops
###############################################图片操作############################################
def resize_image(image,resize_height,resize_width):
    '''
    :param image: rgb image
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image = cv2.resize(image, (resize_width, resize_height))
    return image

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def get_images(image_list,resize_height=0,resize_width=0,whiten=False):
    images = []
    for image_path in image_list:
        # img = misc.imread(os.path.join(images_dir, i), mode='RGB')
        image=read_image(image_path)
        if resize_height > 0 and resize_width > 0:
            image = cv2.resize(image, (resize_width, resize_height))
        if whiten:
            image = prewhiten(image)
        images.append(image)
    images = np.stack(images)
    return images

if __name__=='__main__':
    image_path="../dataset/test_images/huge_5.jpg"
    image=read_image(image_path)
    x=50
    y=50
    w=200
    h=100
    box=[x,y,x+w,y+h]
    show_image_boxes("test_image",image,[box])
    # crop_imgs=crop_images(image,[box])
    # crop_img=crop_imgs[0,:]
    crop_img=crop_image(image,box)
    show_image("crop_img",crop_img)

