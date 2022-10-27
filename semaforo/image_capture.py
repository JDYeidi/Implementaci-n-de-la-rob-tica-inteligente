#!/usr/bin/env python

import os
import rospy
import cv2
import cv_bridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np

class ImageProcessing():
	
	def __init__(self):
		rospy.init_node("image_capture")
		self.rate = rospy.Rate(20)

		rospy.Subscriber("/video_source/raw", Image, self.sourceCallback)
		self.input_image = None
		
		self.bridge = cv_bridge.CvBridge()

		self.filePath = os.path.join("/home/puzzlebot/Documents", "opencvImages")
		self.filePath = os.path.join(self.filePath, "outDir")

		self.writeFrames = True
		self.frameCounter = 0

	def sourceCallback(self, data):
		self.input_image = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'bgr8')

	def writeImage(self, imagePath, inputImage):
		imagePath = imagePath + ".png"
		cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
		print("Wrote Image: " + imagePath)

	def main(self):
		while not rospy.is_shutdown():
			self.rate.sleep()

			if self.input_image is not None:

				if self.writeFrames:
					# Set output path:
					outFileName = "frameOut-" + str(self.frameCounter)
					framePath = os.path.join(self.filePath, outFileName)

					# Write the image:
					self.writeImage(framePath, self.input_image)
					self.frameCounter = self.frameCounter + 1

					# Display the resulting frame
					cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
					cv2.imshow("Video Feed", self.input_image)

		cv2.destroyAllWindows()

if __name__ == '__main__':

	try:
		image_processing = ImageProcessing()
		image_processing.main()

	except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
		pass
