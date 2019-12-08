# coding=utf-8

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import os
from glob import glob
import argparse
import imutils
import cv2
import csv

# return the midpoint of A and B
def midpoint(A, B):
    return ((A[0] + B[0])/2, (A[1] + B[1])/2)

# read plant_ID
def init():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path of the folder")
    ap.add_argument("-i", "--identity", required=True, help="plant_ID")
    args = vars(ap.parse_args())
    plant_ID = args["identity"]
    folder_path = args["path"]
    return plant_ID, folder_path


def extract_pot(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([26, 255, 255])
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_orange = cv2.erode(mask_orange, None, iterations=1)
    mask_orange = cv2.dilate(mask_orange, None, iterations=8)
    mask_orange = cv2.erode(mask_orange, None, iterations=6)
    mask_orange = cv2.dilate(mask_orange, None, iterations=4)
    mask_orange = cv2.erode(mask_orange, None, iterations=5)

    zeros = np.zeros(320)

    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_purple = cv2.erode(mask_purple, None, iterations=1)
    mask_purple = cv2.dilate(mask_purple, None, iterations=3)
    mask_purple = cv2.erode(mask_purple, None, iterations=3)

    mask_all = cv2.add(mask_purple, mask_orange)
    mask_all[522:] = zeros
    mask_all[0:int((4 / 5)*560)] = zeros

    res_all = cv2.bitwise_and(image, image, mask=mask_all)
    return res_all

def calculate_pot_width(res):
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    extLeft = tuple(cnts[0][cnts[0][:, :, 0].argmin()][0])
    extRight = tuple(cnts[0][cnts[0][:, :, 0].argmax()][0])
    pot_width = extRight[0] - extLeft[0]

    return pot_width

def get_zoom_ratio(actual_size, pixel_size):
    zoom_ratio = actual_size / pixel_size
    if zoom_ratio > 0.4:
        return 0.33
    else:
        return zoom_ratio



def calculate_inflorescence_width_and_height(image,zoom_ratio):
    if zoom_ratio < 0.22:
        return 0,0,0,0,0
    # define range of purple color in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])

    # threshold the HSV image to get only purple colors
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    a = np.zeros(320)
    mask[420:] = a
    #cv2.imshow("mask", mask)

    # noise reduction using erosion followed by dilation
    mask = cv2.erode(mask, None, iterations=2)
    #cv2.imshow("mask1", mask)
    mask = cv2.dilate(mask, None, iterations=3)
    #cv2.imshow("mask2", mask)
    res = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow("res", res)

    # convert BGR to GRAY
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # blur the image slightly
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours of the object
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return 0,0,0,0,0

    # sort the contours from left to right
    (cnts,_) = contours.sort_contours(cnts)

    height_width_dic = {}
    width = 0
    height = 0
    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 50:
            continue

        # calculate the bounding rectangle according to the contour of the object
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear in
        # top-left, top-right, bottom-right, and bottom-left order
        # draw the outline of the rectangle
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # compute the midpoint between top-left and top-right coordinates
        # compute the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint coordinates of top-left and top-right and top-right and bottom-right respectively
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between two midpoints
        height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        height_width_dic[tltrY]=[width,height,tltrY,blbrY,blbrX]

        # Draw the result
        '''
        cv2.putText(orig, "{:.1f}".format(width),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        cv2.putText(orig, "{:.1f}".format(height),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        '''

        #cv2.imshow("Image", orig)
        #cv2.waitKey(0)

    if height == 0 and width == 0:
        return 0,0,0,0,0
    else:
        #print(height_width_dic)
        highest_inflorescence = min(height_width_dic.keys())
       # print(highest_inflorescence)
        width = zoom_ratio * height_width_dic[highest_inflorescence][0]
        height = zoom_ratio * height_width_dic[highest_inflorescence][1]
        highest_point = height_width_dic[highest_inflorescence][2]
        lowest_point_Y = height_width_dic[highest_inflorescence][3]
        lowest_point_X = height_width_dic[highest_inflorescence][4]

        return width, height, highest_point, lowest_point_Y,lowest_point_X


def calculate_stem_height(image,inflorescence_bottom_Y,inflorescence_bottom_X,zoom_ratio,flag):
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

    #cv2.imshow("res", res)
    a = np.zeros(320)
    stem_top = []
    stem_bottom = []
    #cv2.imshow("image", image)

    #corp_threshold = int(560 * (4 / 5))
    # print(a)
    #mask[0:corp_threshold] = a

    #mask[550:560] = a

    if flag == 0:
        if zoom_ratio < 0.25:
            mask = cv2.erode(mask, None, iterations=3)
            mask = cv2.dilate(mask, None, iterations=2)
            j = 0
            #cv2.imshow("mask", mask)
            for i in range(560):
                for k in range(320):
                    if mask[i][k] == 255:
                        stem_top = [k, i]
                        #print(stem_top)
                        j = 1
                        break
                if j == 1:
                    break
            stem_bottom = [145, 535]
            stem_height = dist.euclidean((stem_top[0], stem_top[1]), (stem_bottom[0], stem_bottom[1]))
            stem_height = stem_height * zoom_ratio
            #print(stem_height)
            #cv2.waitKey(0)
            return stem_height

        else:
            kernel = np.ones((10, 1), np.uint8)  # 1, 13
            # cv2.imshow("mask2", mask)
            res = cv2.erode(res, None, iterations=1)
            res[0:100] = a
            res[493:] = a
            # print(res.shape)
            #cv2.imshow("res1", res)
            res = cv2.erode(res, kernel, iterations=1)
            #cv2.imshow("res2", res)
            kernel2 = np.ones((10, 1), np.uint8)
            res = cv2.dilate(res, kernel2, iterations=2)
            #cv2.imshow("res2", res)
            j = 0
            for i in range(560):
                for k in range(320):
                    if res[i][k] == 255:
                        stem_top = [k, i]
                        #print(stem_top)
                        j = 1
                        break
                if j == 1:
                    break

            mask2[0:445] = a
            kernel = np.ones((1, 10), np.uint8)
            mask2 = cv2.erode(mask2, kernel, iterations=1)
            # cv2.imshow("mask", mask2)
            mask2 = cv2.dilate(mask2, kernel, iterations=3)
            # cv2.imshow("mask1", mask2)
            cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            print("cnt: %s" % cnts)
            (cnts, _) = contours.sort_contours(cnts)
            stem_bottom_dic = {}
            for c in cnts:
                if cv2.contourArea(c) < 10:
                    continue
                #orig = image.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                midX = (tl[0] + tr[0]) / 2
                midY = (tl[1] + bl[1]) / 2
                stem_bottom = [midX, midY]
                stem_bottom_dic[midY] = stem_bottom

            lowest_bottom = max(stem_bottom_dic.keys())
            stem_bottom = stem_bottom_dic[lowest_bottom]
           # print(stem_bottom)
            #cv2.imshow("orig", orig)
            stem_height = dist.euclidean((stem_top[0], stem_top[1]), (stem_bottom[0], stem_bottom[1]))
            stem_height = stem_height * zoom_ratio
            #print(stem_height)
            #cv2.waitKey(0)
            return stem_height

    else:
        stem_top = [inflorescence_bottom_X, inflorescence_bottom_Y]

        mask2[0:445] = a
        kernel = np.ones((1, 10), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1)
        # cv2.imshow("mask", mask2)
        mask2 = cv2.dilate(mask2, kernel, iterations=3)
        # cv2.imshow("mask1", mask2)
        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        stem_bottom_dic = {}
        for c in cnts:
            if cv2.contourArea(c) < 10:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            midX = (tl[0] + tr[0]) / 2
            midY = (tl[1] + bl[1]) / 2
            stem_bottom = [midX, midY]
            stem_bottom_dic[midY] = stem_bottom

        lowest_bottom = max(stem_bottom_dic.keys())
        stem_bottom = stem_bottom_dic[lowest_bottom]
        #print(stem_bottom)
        #cv2.imshow("orig", orig)
        stem_height = dist.euclidean((stem_top[0], stem_top[1]), (stem_bottom[0], stem_bottom[1]))
        stem_height = stem_height * zoom_ratio
        #print(stem_height)
        #cv2.waitKey(0)
        return stem_height


def calculate_plant_height(image, zoom_ratio):
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
    a = np.zeros(320)
    plant_top = []
    plant_bottom = []
    res = cv2.add(mask2, mask3)
    #cv2.imshow("q",res)
    res = cv2.erode(res, None, iterations=1)
    res[0:100] = a
    res = cv2.dilate(res, None, iterations=1)
    #cv2.imshow("res",res)
    #cv2.waitKey(0)
    j = 0
    for i in range(560):
        for k in range(320):
            if res[i][k] == 255:
                plant_top = [k, i]
                #print(plant_top)
                j = 1
                break
        if j == 1:
            break

    if zoom_ratio < 0.25:
        plant_bottom = [145, 535]
        plant_height = plant_bottom[1]-plant_top[1]
        plant_height = plant_height * zoom_ratio
        #print(plant_height)

        return plant_height

    else:
        mask2[0:445] = a
        kernel = np.ones((1, 10), np.uint8)
        mask2 = cv2.erode(mask2, kernel, iterations=1)
        # cv2.imshow("mask", mask2)
        mask2 = cv2.dilate(mask2, kernel, iterations=3)
        # cv2.imshow("mask1", mask2)
        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        plant_bottom_dic = {}
        for c in cnts:
            if cv2.contourArea(c) < 10:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            midX = (tl[0] + tr[0]) / 2
            midY = (tl[1] + bl[1]) / 2
            plant_bottom = [midX, midY]
            plant_bottom_dic[midY] = plant_bottom

        lowest_bottom = max(plant_bottom_dic.keys())
        plant_bottom = plant_bottom_dic[lowest_bottom]
        #print(plant_bottom)
        #print(plant_top)
        plant_height = plant_bottom[1]-plant_top[1]
        plant_height = plant_height * zoom_ratio
        #print(plant_height)
        #cv2.waitKey(0)
        return plant_height


'''
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        for c in cnts:
            if cv2.contourArea(c) < 50:
                continue
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)  # tlblY is the lowest part of the plant
            height = abs(top - tlblY)
            return height

'''

def output_csv(info_list,plant_ID):
    csv_name = "plant_traits_"+plant_ID+".csv"
    with open(csv_name, mode='w') as csv_file:
        fieldnames = ['plant_ID','date','inflorescence_width', 'inflorescence_height', 'stem_height', 'plant_height']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for info in info_list:
            writer.writerow(info)



if __name__ == "__main__":
    '''
    image = init()
    pot = extract_pot(image)
    pot_width = calculate_pot_width(pot)
    zoom_ratio = get_zoom_ratio(24, pot_width)

    print(zoom_ratio)
    inflorescence_width, inflorescence_height, inflorescence_top, inflorescence_bottom = calculate_inflorescence_width_and_height(image)
    print([inflorescence_width,inflorescence_height,inflorescence_top,inflorescence_bottom])
    #plant_height = calculate_plant_height(image,1,25)#inflorescence_top
    #output_txt_file(inflorescence_width,inflorescence_h
    # eight,plant_height)
    '''
    plant_ID,folder_path = init()
    print('Plant_id: %s , Path: %s' % (plant_ID, folder_path))
    #print(folder_path)
    pattern = folder_path+"/fold3_model_4_300_0."
    pattern = pattern+plant_ID+"*.png"
    l = glob(pattern)
    l_sorted=sorted(l)
    print(l_sorted)
    #print(l_sorted)
    info_list = []
    for image_path in l_sorted:
        flag = 1
        date=((image_path.split('/')[-1]).split('_')[-1]).split('.')[0]
        image = cv2.imread(image_path)
        pot = extract_pot(image)
        pot_width = calculate_pot_width(pot)
        zoom_ratio = get_zoom_ratio(24, pot_width)
        #print(zoom_ratio)
        inflorescence_width, inflorescence_height, inflorescence_top, inflorescence_bottom_Y,inflorescence_bottom_X = calculate_inflorescence_width_and_height(image,zoom_ratio)
        if inflorescence_height == 0 and inflorescence_width == 0:
            flag = 0
        stem_height = calculate_stem_height(image, inflorescence_bottom_Y, inflorescence_bottom_X, zoom_ratio, flag)
        plant_height = calculate_plant_height(image, zoom_ratio)

        info = {'plant_ID':plant_ID,'date':date,'inflorescence_width':inflorescence_width,'inflorescence_height':inflorescence_height,'stem_height':stem_height,'plant_height':plant_height}
        info_list.append(info)
        #print([inflorescence_width, inflorescence_height, inflorescence_top, inflorescence_bottom])
    #cv2.waitKey(0)
    #print(info_list)
    output_csv(info_list,plant_ID)

