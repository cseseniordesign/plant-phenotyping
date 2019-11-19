# coding=utf-8

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# return the midpoint of A and B
def midpoint(A, B):
    return ((A[0] + B[0])/2, (A[1] + B[1])/2)

# read the image
def init():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    # ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    return image

def calculate_inflorescence_width_and_height(image):
    # define range of purple color in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([100, 50, 50])
    upper_purple = np.array([200, 200, 200])
    # threshold the HSV image to get only purple colors
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    # noise reduction using erosion followed by dilation
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(image, image, mask=mask)
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
    # sort the contours from left to right
    (cnts, _) = contours.sort_contours(cnts)
    # initialize 'pixels per metric'
    pixelsPerMetric = None

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

        # draw four vertices of the rectangle
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # compute the midpoint between top-left and top-right coordinates
        # compute the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint coordinates of top-left and top-right and top-right and bottom-right respectively
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between two midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # height
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # width

        '''
        # draw the midpoint of four edges of the rectangle
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # Initialize the measurement index value, the width of the reference object in the picture
        # has been calculated by Euclidean distance, and the actual size of the reference object is known

        # if pixelsPerMetric is None:
        #	pixelsPerMetric = dB / args["width"]

        # Calculate the actual size (width and height) of the target, expressed in feet
        # dimA = dA / pixelsPerMetric
        # dimB = dB / pixelsPerMetric

        # Draw the result
        cv2.putText(orig, "{:.1f}".format(dB),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        cv2.putText(orig, "{:.1f}".format(dA),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        # display
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
        '''
        return dB, dA, int(tltrY)

def calculate_plant_height(image,top):
    # define range of orange color in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([26, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((1, 13), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(image, image, mask=mask)
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
        (tlblX, tlblY) = midpoint(tl, bl)   # tlblY is the lowest part of the plant
        height = abs(top-tlblY)

        return height

def output_txt_file(inflorescence_width,inflorescence_height,plant_height):
    s = "inflorescence width: "+str(inflorescence_width)+"\ninflorescence height: "+str(inflorescence_height)+"\nplant height: "+str(plant_height)
    with open("output.txt", "w") as f:
        f.write(s)

if __name__ == "__main__":
    image = init()
    inflorescence_width, inflorescence_height, inflorescence_top = calculate_inflorescence_width_and_height(image)
    plant_height = calculate_plant_height(image,inflorescence_top)
    print([inflorescence_width,inflorescence_height,plant_height])
    output_txt_file(inflorescence_width,inflorescence_height,plant_height)