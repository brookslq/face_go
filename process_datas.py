# -*- coding: utf-8 -*-
import sys
import os
import cv2
import dlib


'''
对数据进行处理，以达到创建自己的数据集
'''

input_dir = 'input/lfw/'
output_dir = 'other_faces/'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用dlib自带的frontal_face_detector方法提取特征
detector = dlib.get_frontal_face_detector()

index = 1

for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1: y1, x2: y2]
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)

                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                sys.exit(0)
                break


