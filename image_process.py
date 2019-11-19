# coding=utf-8

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# 定义一个中点函数，后面会用到
# define a midpoint function
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# def read_image(image):
	
def extract_stem(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([11,43,46])
	upper_yellow = np.array([26,255,255])
	kernel = np.ones((20,2),np.uint8)
	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	mask = cv2.erode(mask, kernel, iterations=1)
	# 膨胀操作，先腐蚀后膨胀以滤除噪声
	# dilation I think these two steps can filter the noise
	mask = cv2.dilate(mask, kernel, iterations=2)
	mask = cv2.erode(mask, kernel, iterations=2)
	res = cv2.bitwise_and(image,image, mask = mask)
	cv2.imshow('res',res)
	return res

def extract_inflorescence(image):
	# 定义紫色花蕾的颜色范围
	# define the range of purple color
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_purple = np.array([100,50,50])
	upper_purple = np.array([200,200,200])
	
	mask = cv2.inRange(hsv, lower_purple, upper_purple)
	# cv2.imshow('mask', mask)
	# 腐蚀操作
	# erosion
	# 膨胀操作，先腐蚀后膨胀以滤除噪声
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	res = cv2.bitwise_and(image,image, mask = mask)
	cv2.imshow('res',res)
	return res
	
	
def output_result():
	return 0

def init(image):
	# 设置一些需要改变的参数
	# set arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to the input image")
	# ap.add_argument("-w", "--width", type=float, required=True,
	#	help="width of the left-most object in the image (in inches)")
	args = vars(ap.parse_args())
	# 读取输入图片
	# read image
	image = cv2.imread(args["image"])
	return image

def draw_stem(res):
	pass
def draw_inflorescence(res):
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
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		
		# 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
		# Sort the contour points according to the order of top-left, top-right, bottom-right and bottom-left, 
		# and draw the BB of outer tangent, which is represented by green line
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		# 绘制BB的4个顶点，用红色的小圆圈来表示
		# Draw the four vertices of BB, represented by small red circles
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

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
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

		# 在中心点之间绘制直线，用紫红色的线来表示
		# Draw a line between the center points, indicated by a magenta line
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)

		# 计算两个中心点之间的欧氏距离，即图片距离
		# Calculate the Euclidean distance between two center points, that is, the distance of the picture
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
		# Initialize the measurement index value, the width of the reference object in the picture
		#  has been calculated by Euclidean distance, and the actual size of the reference object is known

		# if pixelsPerMetric is None:
		#	pixelsPerMetric = dB / args["width"]

		# 计算目标的实际大小（宽和高），用英尺来表示
		# Calculate the actual size (width and height) of the target, expressed in feet
		# dimA = dA / pixelsPerMetric
		# dimB = dB / pixelsPerMetric
	
		# 在图片中绘制结果
		# Draw the result in the image
		cv2.putText(orig, "{:.1f}".format(dB),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (0, 0, 0), 2)
		cv2.putText(orig, "{:.1f}".format(dA),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (0, 0, 0), 2)
		# 显示结果
		# show result
		cv2.imshow("Image", orig)
		cv2.waitKey(0)

def draw_all(inflorescence,stem):
	draw_inflorescence(inflorescence)
	# draw_stem(stem)
	draw_inflorescence(stem)

def output_result():
	pass

if __name__ == "__main__":
	# py image_process.py --image image.png
	image = np.zeros((1,1,1), np.uint8)
	image = init(image) # read image
	inflorescence = extract_inflorescence(image)
	stem = extract_stem(image)
	draw_all(inflorescence,stem)
	output_result()