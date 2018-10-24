#!/usr/bin/env python
import roslib
roslib.load_manifest('lane_detector')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def frame_publisher():
	rospy.init_node("frame_publisher",anonymous=True)
	pub = rospy.Publisher("image_topic",Image,queue_size=1)
	while(not rospy.is_shutdown()):
		cap = cv2.VideoCapture('road.mp4')
		while(cap.isOpened() and not rospy.is_shutdown()):
			ret,frame = cap.read()
			if(ret!=1):
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			image_message = CvBridge().cv2_to_imgmsg(frame,"bgr8")
			cv_image = CvBridge().imgmsg_to_cv2(image_message,"bgr8")
			pub.publish(image_message)
			rospy.loginfo("lel")
			rospy.Rate(25).sleep()
			#cv2.imshow('frame',cv_image)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	try:
		frame_publisher()
	except rospy.ROSInterruptException:
			pass


