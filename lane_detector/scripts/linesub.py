#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('lane_detector')
import sys
import rospy
from lane_detector.msg import * 

def callback(data):
	rospy.loginfo(data)
		
def line_subscriber():
	rospy.init_node("line_subscriber",anonymous=True)
	rospy.Subscriber('line_topic',Lines,callback)
	rospy.spin()

if __name__ == '__main__':
    line_subscriber()