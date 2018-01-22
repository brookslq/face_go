# -*- coding: utf-8 -*-

'''

通过Keras创建的预测人脸是否是自己

'''


import cv2
import sys
import gc
import dlib
from train_cnn_model import Model

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    #     sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model(file_path='model/me.face.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = "face_opencv_model/haarcascade_frontalface_alt2.xml"

    # 使用dlib进行人脸检测
    detector = dlib.get_frontal_face_detector()

    # 循环检测识别人脸
    while True:
        _, frame = cap.read()  # 读取一帧视频

        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 基于dlib检测
        dets = detector(frame_gray, 1)
        if not len(dets):
            cv2.imshow('img', frame)
            # # 等待10毫秒看是否有按键输入
            # k = cv2.waitKey(10)
            # # 如果输入q则退出循环
            # if k & 0xFF == ord('q'):
            #     break
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = frame[x1:y1, x2:y2]
            # 调整图片的尺寸，获取人脸
            face = cv2.resize(face, (64, 64))

            # 截取脸部图像提交给模型识别这是谁
            faceID = model.face_predict(face)

            # 如果是“我”
            if faceID == 0:
                # 文字提示是谁
                cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 3)
                cv2.putText(frame, 'BrooksL',
                            (x1, y1),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽
                cv2.imwrite('isBrooksL.jpg', frame)

            else:
                # 文字提示是谁
                cv2.rectangle(frame, (x2, x1), (y2, y1), (0, 255, 0), 3)
                cv2.putText(frame, 'NoBody',
                            (x1, y1),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (0, 255, 255),  # 颜色
                            2)  # 字的线宽
                cv2.imwrite('notBrooksL.jpg',frame)

            cv2.imshow('image', frame)

        # 基于opencv检测
        # # 使用人脸识别分类器，读入分类器
        # cascade = cv2.CascadeClassifier(cascade_path)
        #
        # # 利用分类器识别出哪个区域为人脸
        # faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        # if len(faceRects) > 0:
        #     for faceRect in faceRects:
        #         x, y, w, h = faceRect
        #
        #         # 截取脸部图像提交给模型识别这是谁
        #         image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
        #         faceID = model.face_predict(image)
        #
        #         # 如果是“我”
        #         if faceID == 0:
        #             cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
        #
        #             # 文字提示是谁
        #             cv2.putText(frame, 'BrooksL',
        #                         (x + 30, y + 30),  # 坐标
        #                         cv2.FONT_HERSHEY_SIMPLEX,  # 字体
        #                         1,  # 字号
        #                         (255, 0, 255),  # 颜色
        #                         2)  # 字的线宽
        #         else:
        #             # 文字提示是谁
        #             cv2.putText(frame, 'NoBody',
        #                         (x + 30, y + 30),  # 坐标
        #                         cv2.FONT_HERSHEY_SIMPLEX,  # 字体
        #                         1,  # 字号
        #                         (0, 255, 255),  # 颜色
        #                         2)  # 字的线宽
        #             pass

        # cv2.imshow("识别李强", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()