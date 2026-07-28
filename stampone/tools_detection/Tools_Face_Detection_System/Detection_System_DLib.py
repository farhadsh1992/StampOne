
"""
@--23.02.2023--@
Author: github/farhadsh1992
INFO:
    Dlib CNN face detector: locates a face's bounding box for cropping.
LAST_UPDATE:
	-  http://dlib.net/face_landmark_detection.py.html
	- https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
"""



import cv2
import os
import numpy as np
import glob
import dlib



class landmark_face_dlib():
	def __init__(self):
    	prefix = "./"
    	detector_path = os.path.join(prefix, './Tools_Face_Detection_System/net-data/mmod_human_face_detector.dat')
    	self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)

    def call(self, image)
    	detected_faces = self.face_detector(image, 1)
    	d = detected_faces[0].rect
    	left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()

    	#d_coped =  np.array([top,left, bottom - top,right -left],dtype=np.int32)
    	point_croped =  np.array([top,left, bottom,right],dtype=np.int32)
   	 	#d_coped = np.array([top,left, 180,180],dtype=np.int32)
   	 	croped_face = crop_image(image, crop=point_coped)

    	return croped_face, point_croped