import cv2
import numpy as np
img = cv2.imread('p2.png')
img1 = img
img = img[0:480, 80:244]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_purple = np.array([0,0,0])
upper_purple = np.array([255,255,254])
mask = cv2.inRange(hsv, lower_purple, upper_purple,)
kernel = np.ones((2,2),np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=10)
mask = cv2.erode(mask, kernel, iterations=1)
cv2.imshow('res',mask)
cv2.waitKey(0)
res = cv2.bitwise_and(img,img, mask = mask)
cv2.imshow('res',res)
cv2.waitKey(0)
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
cv2.imshow('res',gray)
cv2.waitKey(0)

ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
cv2.imshow('res',binary)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
cv2.drawContours(img,contours,0,(0,0,255),3)
 
cv2.imshow("img", img)
cv2.waitKey(0)
plant = contours[0]
leftmost = tuple(plant[:,0][plant[:,:,1].argmax()])
rightmost = tuple(plant[:,0][plant[:,:,1].argmin()])
cv2.circle(img1, leftmost, 2, (0,0,0),4) 
cv2.circle(img1, rightmost, 2, (0,0,0),4) 

cv2.imshow("img", img1)
cv2.waitKey(0)