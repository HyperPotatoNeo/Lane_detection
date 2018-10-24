#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('lane_detector')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from lane_detector.msg import * 
from cv_bridge import CvBridge, CvBridgeError


def draw_lines(
    img,
    lines,
    color=[0xFF, 0, 0],
    thickness=10,
    ):
    #rospy.loginfo(lines)

    imshape = img.shape
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = []
    i = 0
    j = 0
    y_min = img.shape[0]
    y_max = int(img.shape[0] * 0.611)
    for line in lines:
        for (x1, y1, x2, y2) in line:
            if (y2 - y1) / (x2 - x1) < 0:

              
                i += 1
                left_x1.append(x1)
                left_x2.append(x2)
            else:

       

                j += 1

            
                right_x1.append(x2)
                right_x2.append(x1)

        #rospy.loginfo(str(i) + ' ' + str(j))

 
        if len(left_x1) == 0:
            left_x1.append(0)
        if len(left_x2) == 0:
            left_x2.append(0)
        if len(right_x1) == 0:
            right_x1.append(0)
        if len(right_x2) == 0:
            right_x2.append(0)
    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(np.nanmean(right_x1))
    r_avg_x2 = np.int(np.nanmean(right_x2))
    m1=float(y_max-y_min)/(l_avg_x2-l_avg_x1)
    m2=float(y_max-y_min)/(r_avg_x2-r_avg_x1)
    rospy.loginfo(str(-m1)+" "+str(-m2))
    c1=y_max+m1*l_avg_x2
    c2=y_max+m2*r_avg_x2
    rospy.loginfo(str(c1)+" "+str(c2))
    pub = rospy.Publisher("line_topic",Lines,queue_size=1)
    lel = Lines()
    lel.m1=m1
    lel.m2=m2
    lel.c1=c1
    lel.c2=c2
    pub.publish(lel)


    cv2.line(img, (l_avg_x1, y_min), (l_avg_x2, y_max), color,
             thickness)
    cv2.line(img, (r_avg_x1, y_min), (r_avg_x2, y_max), color,
             thickness)


def hough_lines(
    img,
    rho,
    theta,
    threshold,
    min_line_len,
    max_line_gap,
    ):
   

    lines = []
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
        )
    line_img = np.zeros(img.shape, dtype=np.uint8)
    if len(lines) != 0:

        draw_lines(line_img, lines)
        return line_img
        return -1


def region_of_interest(img, vertices):
   
    mask = np.zeros_like(img)

   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (0xFF, ) * channel_count
    else:
        ignore_mask_color = 0xFF

 
    cv2.fillPoly(mask, vertices, ignore_mask_color)

   
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def callback(data):

    cv_image = CvBridge().imgmsg_to_cv2(data, 'bgr8')

   
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    lower_yellow = np.array([20, 100, 100], dtype='uint8')
    upper_yellow = np.array([30, 0xFF, 0xFF], dtype='uint8')
    mask_yellow = cv2.inRange(cv_image, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 0xFF)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    canny = cv2.Canny(gray_image, 50, 150)
    poly = np.array([[[0, 245], [160, 220], [251, 212], [474, 274], [0,317]]])
    roi = region_of_interest(canny, poly)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    lel = hough_lines(
        roi,
        1,
        1 * np.pi / 180,
        20,
        1,
        100,
        )

    try:
        cv2.imshow('frame', canny)
        cv2.imshow('fram2', lel)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            pass
    except:
        pass


def lane_publisher():
    rospy.init_node('lane_publisher', anonymous=True)
    rospy.Subscriber('image_topic', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    lane_publisher()

			