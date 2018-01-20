import cv2
import dlib
import os
import sys
import random

output_dir = 'my_face/'
size = 98

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 改变图片的亮度对比度等等
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i , c] = tmp

    return img

# 使用dlib,进行人像检测并截图保存
def catch_face():
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    index = 1
    while True:
        # 需要准备1w张头像
        if (index <= 10000):
            if index % 1000 == 0:
                print("已经存到了第：%d 张" % index)

            ret, img = cap.read()
            # 需要先转换为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片的对比度亮度，随机分配，增加样本的多样性
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)
                cv2.imwrite(output_dir + "/" + str(index) + ".jpg", face)
                index += 1

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cap.release()
                break

        else:
            print("Finished")
            break


if __name__ == '__main__':
    catch_face()