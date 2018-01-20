#-*- coding: utf-8 -*-

import cv2
# import sys
# from PIL import Image


'''
通过opencv进行的人脸检测，并创建了自己的数据集
'''

def catch_face_from_camera(window_name, camera_idx, catch_pic_num, path_name):
    # 从摄像头获取视频
    cap = cv2.VideoCapture(camera_idx)

    # 初始化人脸识别分类器
    classfier = cv2.CascadeClassifier("face_opencv_modle/haarcascade_frontalface_alt2.xml")
    color = (255, 0, 0)
    num = 0


    # 获取视频播放界面长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 10)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 10)

    # 定义编码器 创建 VideoWriter对象q
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    while(cap.isOpened()):
        # 读取帧摄像头
        ret, frame = cap.read()

        # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测， 1.2 和 2 分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=2, minSize=(72, 72))
        if len(faceRects)>0:
            # 框出每个人脸
            for faceRect in faceRects:
                x, y, w, h = faceRect


                # 超出保存最大值则跳出
                if num > catch_pic_num:
                    break
                # 画出矩形框
                cv2.rectangle(frame, (x-10, y-10,), (x+w+10, y+h+10), color, 2)
                # 显示当前捕捉到了多少人脸图片
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num: %d' % (num), (x+30, y+30), font, 1, (255,0,255), 4)

        # 如果超过最大值则保存数量结束程序
        if num > catch_pic_num: break

        if ret == True:
            # 输出当前帧
            out.write(frame)
            cv2.imshow(window_name, frame)
            # 键盘按 q 退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break


# 主入口
if __name__ == '__main__':
    catch_face_from_camera("李强的人脸识别第一步",0, 20, 'lq')
