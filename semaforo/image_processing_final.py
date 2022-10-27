#!/usr/bin/env python

import rospy
import cv2
import cv_bridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np

class ImageProcessing():
	
	def __init__(self):

		# Inicializar Nodo
		rospy.init_node("image_processing")
		self.rate = rospy.Rate(20)

		# Subscriptores
		rospy.Subscriber("/video_source/raw", Image, self.sourceCallback)
		self.input_image = None

		# Publicadores
		self.image_pub = rospy.Publisher("/output_image", Image, queue_size = 1)

		self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
		self.robot_cmd = Twist()
		
		# Otras Variables
		self.bridge = cv_bridge.CvBridge()

		# Rangos de ROJO 
		self.lower_red_1 = np.array([0,   80,  130])
		self.upper_red_1 = np.array([12,  255, 255])
		self.lower_red_2 = np.array([170, 80,  130])
		self.upper_red_2 = np.array([180, 255, 255])

		# Rangos de AMARILLO
		self.lower_yellow = np.array([16, 60,  130])
		self.upper_yellow = np.array([32, 255, 255])

		# Rangos de VERDE
		self.lower_green = np.array([40, 20,  115])
		self.upper_green = np.array([85, 255, 255])

		# Kernel
		self.kernel = np.ones((4, 4), np.uint8)

		# Parametros del detector
		self.params = cv2.SimpleBlobDetector_Params()
		self.params.filterByCircularity = True
		self.params.minCircularity = 0.70

		self.params.filterByArea = True;
		self.params.minArea = 1500;
		self.params.maxArea = 30000;

		# Blob
		self.detector = cv2.SimpleBlobDetector_create(self.params)

		self.v = 0

	# Input Image Callback
	def sourceCallback(self, data):
		self.input_image = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'bgr8')

	# Funcion de pre procesado para la imagen
	def imProcessing(self, color_result):

		gray = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
		# Eliminacion del fondo y ruido
		ret, simple_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
		dilation = cv2.dilate(simple_thresh, self.kernel, iterations = 1)

		return dilation

	# Funcion para la deteccion de rojo	
	def redDetector(self, hsv):

		# Obtencion de la mascara de color
		red_mask_1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
		red_mask_2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
		red_mask = red_mask_1 + red_mask_2
		red_result = cv2.bitwise_and(self.input_image, self.input_image, mask = red_mask)

		dilation = self.imProcessing(red_result)

		# Deteccion de circulos de acuerdo al color
		keypoints = self.detector.detect(dilation)
		
		if len(keypoints) > 0:
			color = True
		else:
			color = False

		return color, keypoints, red_mask

	# Funcion para la deteccion de amarillo
	def yellowDetector(self, hsv):

		yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
		yellow_result = cv2.bitwise_and(self.input_image, self.input_image, mask = yellow_mask)

		dilation = self.imProcessing(yellow_result)

		keypoints = self.detector.detect(dilation)

		if len(keypoints) > 0:
			color = True
		else:
			color = False

		return color, keypoints, yellow_mask
	
	# Funcion para la deteccion de verde		
	def greenDeetector(self,hsv):

		green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
		green_result = cv2.bitwise_and(self.input_image, self.input_image, mask = green_mask)

		dilation = self.imProcessing(green_result)

		keypoints = self.detector.detect(dilation)

		if len(keypoints) > 0:
			color = True
		else:
			color = False

		return color, keypoints, green_mask

	def main(self):
		while not rospy.is_shutdown():
			self.rate.sleep()

			if self.input_image is not None:
				# Conversion del formato de color
				hsv = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2HSV)
				# Obtencion de las mascaras y circulos detectados de acuerdo al color
				redColor, redKeyp, red_mask = self.redDetector(hsv)
				yellowColor, yellowKeyp, yellow_mask = self.yellowDetector(hsv)
				greenColor, greenKeyp, green_mask = self.greenDeetector(hsv)

				print("Color rojo = " + str(redColor))
				print("Color amarillo = " + str(yellowColor))
				print("Color verde = " + str(greenColor))

				# Condiciones de avance para el Puzzlebot
				if redColor:
					self.v = 0.0
				elif yellowColor:
					self.v = 0.15
				elif greenColor:
					self.v = 0.3

				print(self.v)

				# Publicacion de la velocidad
				self.robot_cmd.linear.x = self.v
				self.cmd_pub.publish(self.robot_cmd)
				self.rate.sleep()

				# Obtencion de la mascara que combina rojo, amarillo y verde
				general_mask = red_mask + yellow_mask + green_mask
				general_result = cv2.bitwise_and(self.input_image, self.input_image, mask = general_mask)

				dilation = self.imProcessing(general_result)

				# Deteccion de circulos rojos, amarillos y verdes
				general_keypoints = redKeyp + yellowKeyp + greenKeyp
				im_with_keypoints = cv2.drawKeypoints(dilation, general_keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

				# Publicacion de la imagen procesada
				output_image = self.bridge.cv2_to_imgmsg(im_with_keypoints, encoding = 'bgr8')	
				self.image_pub.publish(output_image)

		cv2.destroyAllWindows()

if __name__ == '__main__':

	try:
		image_processing = ImageProcessing()
		image_processing.main()

	except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
		pass
