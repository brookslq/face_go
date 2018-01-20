import dlib
import cv2
import random
import sys
import numpy
from skimage import io
from PIL import Image
import time

'''
通过dlib库，进行的人脸检测并输出
'''


RECT_COLOR = (255, 0, 0)


size = 128
output_dir = 'dataset/'

def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


def detector_face(frame):
    """
    检测人脸并输出
    :param frame: 视频截屏图像
    :return: 检测出人脸的图像
    """

    # 使用dlib自带的检测器
    detector = dlib.get_frontal_face_detector()
    # 将当前帧转换成灰度图像
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 进行人脸检测
    dects = detector(frame, 1)

    # 对检测出的模型进行计算
    for i, rect in enumerate(dects):
        # 读取人脸区域坐标
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        # print("脸部坐标：(%d, %d), (%d, %d)" % (left, top, right, bottom))
        # 使用opencv中的函数绘制出人脸的方框 （或者，dlib中也有自带的方法画出人脸）
        cv2.rectangle(frame, (left, top), (right, bottom), RECT_COLOR, 2)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # # out.write(frame)
        # cv2.imshow("Video", frame)

        # 返回检测出人脸的图像
    return frame

def img_from_video():

    # opencv加载视频文件
    # cap = cv2.VideoCapture('*.mp4')
    # 获取摄像头
    capture = cv2.VideoCapture(0)
    # 获取视频播放界面长宽
    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 10)
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 10)

    index = 0

    # 定义编码器 创建 VideoWriter对象q
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
    while (capture.isOpened()):
        ret, frame = capture.read()

        if frame is None:
            break
        else:

            # start = time.clock()
            frame = detector_face(frame)
            # elapsed = (time.clock() - start)
            # print("Time used:", elapsed)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face = relight(frame, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size, size))
            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
            index += 1
            print(index)

            # # out.write(frame)
            cv2.imshow("Face_Go", frame)

        if index == 10:
            capture.release()
            break

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            capture.release()
            break

if __name__ == '__main__':
    img_from_video()

