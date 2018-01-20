# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2

'''
加载数据集
'''

size = 64

# 重新设置图像尺寸，保证数据图片尺寸一统
def resize_iamge(image, height=size, width=size):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape
    # 对于长度不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多少像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        # //表示整除符号
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # rgb 颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，CV2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=True)
    return cv2.resize(constant, (height, width))

# 读取训练数据
images = []
labels = []
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        # 如果是文件夹，继续递归调用
        if os.path.isdir(full_path):
            read_path(full_path)
        # 文件
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_iamge(image, size, size)
                images.append(image)
                labels.append(path_name)

    return images, labels

# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 将输入的所有图片转成四维数组，尺寸为（图片数量*size*size*3)
    # 图片resize之后为64*64像素*3（RGB）
    images = np.array(images)

    # 标注数据， 'my_face'文件夹下是自己的图片，全部指定为0，另外一个文件夹下是其他人的图片，全部指定为1
    labels = np.array([0 if label.endswith('my_face') else 1 for label in labels])

    return images, labels

if __name__ == '__main__':
    # if len(sys.argv) != 0:
    #     print("Usage:%s path_name\r\n" % (sys.argv[0]))
    # else:
    images, labels = load_dataset("/Users/brooks/Desktop/PythonLearn/deeplearning/face_go/data")
