#!/usr/bin/env python

import rospy
import cv2
import cv_bridge
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np

class ImageProcessing():
	
	def __init__(self):

		# Inicializar Nodo
		rospy.init_node("line_following")
		self.rate = rospy.Rate(20)

		rospy.on_shutdown(self.endCallback)

		# Subscriptores
		rospy.Subscriber("/video_source/raw", Image, self.sourceCallback)
		self.input_image = None

		# Publicadores
		self.image_pub = rospy.Publisher("/output_image", Image, queue_size = 1)

		self.c_image_pub = rospy.Publisher("/output_image/cropped_binarized", Image, queue_size = 1)
		self.b_image_pub = rospy.Publisher("/output_image/binarized", Image, queue_size = 1)

		self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
		self.robot_cmd = Twist()

		# Otras Variables
		self.bridge = cv_bridge.CvBridge()

		# Kernel
		self.kernel = np.ones((5, 5), np.uint8)

		# Bandera de deteccion de la linea
		self.line = False

		# Ganancia del controlador
		self.k = 0.002

	# Callback para detener al robot
	def endCallback(self):
		self.robot_cmd.linear.x = 0.0
		self.robot_cmd.angular.z = 0.0
		self.cmd_pub.publish(self.robot_cmd)

	# Input Image Callback
	def sourceCallback(self, data):
		self.input_image = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'bgr8')

	def imROI(self):

		cropped_image = self.input_image[160:240, 0:426]
		cropped_image = self.imProcessing(cropped_image)

		return cropped_image

	# Funcion de pre procesado para la imagen
	def imProcessing(self, image):

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		medBlur = cv2.medianBlur(gray, 5)
		__, th1 = cv2.threshold(medBlur, 90, 255, cv2.THRESH_BINARY_INV)
		closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, self.kernel)

		return closing

	# Deteccion de contornos
	def imContours(self):

		cropped_image = self.imROI()
		drawing = self.input_image
		drawing_binarized = self.imProcessing(self.input_image) 

		# Encontrar contornos
		contours, __ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for c in contours:
			# Obtencion del area
			area = cv2.contourArea(c) 
			print(area) 
			if area > 1500:
				self.line = True
				# Momento
				mu = cv2.moments(c)
				# Coordenadas del centroide
				xPoint = int(mu['m10'] / mu['m00'])
				yPoint = int(mu['m01'] / mu['m00'])
				yFixedPoint = yPoint + 130
				# Dibujo de circulo y rectangulo 
				cv2.circle(drawing, (xPoint, yFixedPoint), 2, (0, 255, 0), -1, 8, 0)
				cv2.rectangle(drawing, (xPoint - 15, yFixedPoint - 10), (xPoint + 15, yFixedPoint + 10), (0, 0, 255), 1, 8, 0)
				cv2.circle(drawing_binarized, (xPoint, yFixedPoint), 2, (0, 255, 0), -1, 8, 0)
				cv2.rectangle(drawing_binarized, (xPoint - 15, yFixedPoint - 10), (xPoint + 15, yFixedPoint + 10), (0, 0, 255), 1, 8, 0)
			#else:
				#self.line = False

		return xPoint, drawing, drawing_binarized

	# Funcion de control
	def controller(self):
		xPoint, dwg, dwg_b = self.imContours()

		if self.line:
			# Calculo del error (centro en X de la imagen - centro en X de la linea)
			error = self.input_image.shape[1]/2 - xPoint
			v = 0.2
			w = self.k*error
		#else:
			#v = 0.0
			#w = 0.0

		self.robot_cmd.linear.x = v
		self.robot_cmd.angular.z = w
		self.cmd_pub.publish(self.robot_cmd)

	def main(self):
		while not rospy.is_shutdown():
			self.rate.sleep()

			if self.input_image is not None:
				
				cropped_image = self.imROI()
				__, drawing, drawing_binarized = self.imContours()
				self.controller()
				
				# Publicacion de la imagen procesada
				output_image = self.bridge.cv2_to_imgmsg(drawing, encoding = 'bgr8')	
				self.image_pub.publish(output_image)

				c_output_image = self.bridge.cv2_to_imgmsg(cropped_image, encoding = '8UC1')	
				self.c_image_pub.publish(c_output_image)

				b_output_image = self.bridge.cv2_to_imgmsg(drawing_binarized, encoding = '8UC1')	
				self.b_image_pub.publish(b_output_image)

if __name__ == '__main__':

	try:
		image_processing = ImageProcessing()
		image_processing.main()

	except (rospy.ROSInterruptException, rospy.ROSException("topic was closed during publish()")):
		pass