#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# define a midpoint function

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def extract_pot(image, height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([11, 43, 46])
    upper_yellow = np.array([26, 255, 255])
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv2.erode(mask_yellow, None, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=8)
    mask_yellow = cv2.erode(mask_yellow, None, iterations=6)
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=4)
    mask_yellow = cv2.erode(mask_yellow, None, iterations=5)

    zeros = np.zeros(get_width(image))

    # mask_yellow[height-40:height] = zeros
    # res_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
    # cv2.imshow('res_yellow', res_yellow)

    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_purple = cv2.erode(mask_purple, None, iterations=1)
    mask_purple = cv2.dilate(mask_purple, None, iterations=3)
    mask_purple = cv2.erode(mask_purple, None, iterations=3)
    res_purple = cv2.bitwise_or(image, image, mask=mask_purple)

    # cv2.imshow('res_purple', res_purple)

    mask_all = cv2.add(mask_purple, mask_yellow)
    mask_all[height - 38:height] = zeros
    mask_all[0:int(4 / 5 * height)] = zeros

    # mask_all = cv2.erode(mask_all, None, iterations=1)

    res_all = cv2.bitwise_and(image, image, mask=mask_all)

    # cv2.imshow('res_all', res_all)

    return res_all


def extract_inflorescence(image):

    # define the range of purple color

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # cv2.imshow('mask', mask)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(image, image, mask=mask)

    # cv2.imshow('res',res)

    return res


def init(image):

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    # ....help="path to the input image")
    # ap.add_argument("-w", "--width", type=float, required=True,
    # ....help="width of the left-most object in the image (in inches)")
    # args = vars(ap.parse_args())

    image = cv2.imread('image3.png')
    return image


def draw_inflorescence(res, zoom_ratio):

    # 输入图片灰度化
    # gray scale

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # 对灰度图片执行高斯滤波
    # Gaussian filter

    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 对滤波结果做边缘检测获取目标
    # detect the edge

    edged = cv2.Canny(gray, 50, 100)

    # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
    # close the gap between edges

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 在边缘图像中寻找物体轮廓（即物体）
    # find contour of the object

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 对轮廓按照从左到右进行排序处理
    # sort the contour from left to right

    (cnts, _) = contours.sort_contours(cnts)

    # initialize 'pixels per metric'

    pixelsPerMetric = None

    # 循环遍历每一个轮廓
    # Loop through each contour

    for c in cnts:

        # 如果当前轮廓的面积太少，认为可能是噪声，直接忽略掉
        # If the area of the current contour is too small, consider it may be noise, and ignore it

        if cv2.contourArea(c) < 50:
            continue

        # 根据物体轮廓计算出外切矩形框
        # Calculate the outcut rectangle according to the contour of the object

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = \
            (cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
        box = np.array(box, dtype='int')

        # 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
        # Sort the contour points according to the order of top-left, top-right, bottom-right and bottom-left,
        # and draw the BB of outer tangent, which is represented by green line

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype('int')], -1, (0, 0, 255), 1)

        # 绘制BB的4个顶点，用红色的小圆圈来表示
        # Draw the four vertices of BB, represented by small red circles

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 1, (0, 0, 255), -1)

        # 分别计算top-left 和top-right的中心点和bottom-left 和bottom-right的中心点坐标
        # Calculate the center point coordinates of top-left
        # and top-right and bottom-left and bottom-right respectively

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # 分别计算top-left和top-right的中心点和top-righ和bottom-right的中心点坐标
        # Calculate the center point coordinates of top-left and top-right and top-righ and bottom-right respectively

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # 绘制BB的4条边的中心点，用蓝色的小圆圈来表示
        # Draw the center point of the four edges of BB, represented by a small blue circle
        # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # 在中心点之间绘制直线，用紫红色的线来表示
        # Draw a line between the center points, indicated by a magenta line
        # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        # ....(255, 0, 255), 2)
        # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        # ....(255, 0, 255), 2)

        # 计算两个中心点之间的欧氏距离，即图片距离
        # Calculate the Euclidean distance between two center points, that is, the distance of the picture

        height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
        # Initialize the measurement index value, the width of the reference object in the picture
        #  has been calculated by Euclidean distance, and the actual size of the reference object is known

        # if pixelsPerMetric is None:
        # ....pixelsPerMetric = dB / args["width"]

        # 计算目标的实际大小（宽和高），用英尺来表示
        # Calculate the actual size (width and height) of the target, expressed in feet

        real_height = round(height * zoom_ratio, 2)
        real_width = round(width * zoom_ratio, 2)

        # 在图片中绘制结果
        # Draw the result in the image

        cv2.putText(
            orig,
            '{:.1f}'.format(real_width),
            (int(tltrX - 15), int(tltrY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            )
        cv2.putText(
            orig,
            '{:.1f}'.format(real_height),
            (int(trbrX + 10), int(trbrY)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            )

        # 显示结果
        # show result

        cv2.imshow('Orig', orig)
        cv2.waitKey(0)
    return (real_height, real_width)


def output_result(inf_width, inf_height, plant_height):
    s = 'inflorescence width: ' + str(inf_width) \
        + ' cm\ninflorescence height: ' + str(inf_height) \
        + ' cm\nplant height: ' + str(plant_height) + ' cm'
    with open('output.txt', 'w') as f:
        f.write(s)


def get_height(image):
    return image.shape[0]


def get_width(image):
    return image.shape[1]


def get_lowest_height(image):

    # height, width in image

    height = image.shape[0]

    # width = image.shape[1]
    # print(height, width)
    # cut the lower part to remove purple noise

    lowest_height = int(4 / 5 * height)

    # print(lowest_height)
    # cv2.line(image, (0,lowest_height), (width,lowest_height),(255, 0, 255), 2)

    return lowest_height


def get_zoom_ratio(actual_size, pixel_size):
    return actual_size / pixel_size


def draw_pot_d(res):

    # gray scale

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Gaussian filter

    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # detect the edge

    edged = cv2.Canny(gray, 50, 100)

    # close the gap between edges

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contour of the object

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contour from left to right

    (cnts, _) = contours.sort_contours(cnts)

    orig = image.copy()
    extLeft = tuple(cnts[0][cnts[0][:, :, 0].argmin()][0])
    extRight = tuple(cnts[0][cnts[0][:, :, 0].argmax()][0])

    # print(extLeft)
    # print(extRight)

    pot_width = extRight[0] - extLeft[0]
    cv2.line(orig, (extLeft[0], extLeft[1]), (extRight[0], extLeft[1]),
             (0, 0, 255), 2)
    cv2.putText(
        orig,
        '{:.1f}'.format(pot_width),
        (int(get_width(orig) / 2), get_lowest_height(orig)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        )

    # cv2.imshow("Image", orig)
    # cv2.waitKey(0)

    return pot_width


def get_plant_height(image, zoom_ratio):
    image_height = get_height(image)
    image_width = get_width(image)

    # define range of orange color in HSV

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([26, 255, 255])
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])
    lower_green = np.array([30, 200, 100])
    upper_green = np.array([100, 255, 190])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask2 = cv2.inRange(hsv2, lower_green, upper_green)
    mask3 = cv2.inRange(hsv3, lower_purple, upper_purple)
    res = cv2.add(mask, mask3)

    cv2.imshow('res', res)
    a = np.zeros(get_width(image))
    stem_top = []
    stem_bottom = []
    cv2.imshow('image', image)

    # corp_threshold = int(560 * (4 / 5))
    # print(a)
    # mask[0:corp_threshold] = a
    # mask[550:560] = a

    if zoom_ratio < 0.25:
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=2)
        j = 0
        cv2.imshow('mask', mask)
        for i in range(image_height):
            for k in range(image_width):
                if mask[i][k] == 255:
                    stem_top = [k, i]

                    # print(stem_top)

                    j = 1
                    break
            if j == 1:
                break
        stem_bottom = [145, 535]
        orig = image.copy()
    else:

        kernel = np.ones((10, 1), np.uint8)  # 1, 13

        # cv2.imshow("mask2", mask)

        res = cv2.erode(res, None, iterations=1)
        res[0:100] = a
        res[493:image_height] = a

        # print(res.shape)

        cv2.imshow('res1', res)
        res = cv2.erode(res, kernel, iterations=1)
        cv2.imshow('res2', res)
        kernel2 = np.ones((10, 1), np.uint8)
        res = cv2.dilate(res, kernel2, iterations=2)
        cv2.imshow('res2', res)
        j = 0
        for i in range(image_height):
            for k in range(image_width):
                if res[i][k] == 255:
                    stem_top = [k, i]

                    # print(stem_top)

                    j = 1
                    break
            if j == 1:
                break

        mask2[0:445] = a
        kernel = np.ones((1, 10), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1)

        # cv2.imshow("mask", mask2)

        mask2 = cv2.dilate(mask2, kernel, iterations=2)

        # cv2.imshow("mask1", mask2)

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < 10:
                continue
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = \
                (cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            box = np.array(box, dtype='int')
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype('int')], -1, (0, 255,
                             0), 2)
            (tl, tr, br, bl) = box

            midX = (tl[0] + tr[0]) / 2
            midY = (tl[1] + bl[1]) / 2
            stem_bottom = [midX, midY]

    plant_height = dist.euclidean((stem_top[0], stem_top[1]),
                                  (stem_bottom[0], stem_bottom[1]))
    real_plant_height = plant_height * zoom_ratio
    real_plant_height = round(real_plant_height, 2)
    cv2.line(orig, (int(stem_top[0]), int(stem_top[1])),
             (int(stem_bottom[0]), int(stem_bottom[1])), (0, 0, 255), 2)
    cv2.putText(
        orig,
        '{:.2f}'.format(real_plant_height),
        (int(stem_top[0]), int(stem_top[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        )
    cv2.imshow('orig', orig)
    cv2.waitKey(0)
    return real_plant_height


if __name__ == '__main__':

        # read image

    image = np.zeros((1, 1, 1), np.uint8)

    # image = init(image)

    image = cv2.imread('image3.png')
    flowered = True

    # get zoom ratio

    pot = extract_pot(image, get_height(image))
    pot_d = draw_pot_d(pot)
    zoom_ratio = get_zoom_ratio(24, pot_d)
    print(zoom_ratio)

    if flowered == True:
        inflorescence = extract_inflorescence(image)
        (inf_height, inf_width) = draw_inflorescence(inflorescence,
                zoom_ratio)
    else:
        (inf_height, inf_width) = (0, 0)

    # stem = extract_stem(image)
    # height = get_height(image)
    # draw_all(inflorescence,stem, zoom_ratio)

    plant_height = get_plant_height(image, zoom_ratio)

    # print(plant_height)
    # print(inf_height, inf_width)

    cv2.waitKey(0)
    output_result(inf_width, inf_height, plant_height)
