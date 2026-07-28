"""
Rotates/aligns a face image level using MobileNet landmark detection
(eye and mouth-corner points) before it is fed to the face detector.
"""

import cv2
from stampone.FarhadCV.Tools import read_files, read_folders, estimater, tcolors
from FaceDecMobilenets import SSDFaceDetector, LandmarkDetector

import numpy as np
import matplotlib.pyplot as plt

#from FSSDecoderModels import FSSDecoder









class FaceAlignment():
    def __init__(self, model_direction = "./MobilenetModels/mobilenet_fp32_cpu.tflite"):
        
        self.LD =  LandmarkDetector(model_direction)
        
    def fitOneShot(self, image, angle = 0):
        
        height, width, _ = image.shape 
        self.angle = angle
        landmarksold = self.LD.predict(image)
        #find center of the face
        landmarks = [[]]
        for ith, land in enumerate(landmarksold[0]):
            landmarks[0].append([])
            landmarks[0][ith].append(land[1] * height)
            landmarks[0][ith].append(land[0] * width)
        center_of_face = landmarks[0][2]
        center_distance = np.sqrt((width/2 - center_of_face[0])**2 + (height/2 - center_of_face[1])**2)
        
        l_dist = height *  width
    
        if(l_dist > center_distance):
            l_dist = center_distance       
    
    
        face_scale=1.1
        face_dim = self.find_face_dim(landmarks, height, width) * face_scale
        
        ew_bounding_box=[center_of_face[0]-face_dim/2,center_of_face[1]-face_dim/2, face_dim, face_dim]
    
        #get rotation angle
        if self.angle == 0:
            scale=0.7
        else:
            scale=1.2
        trotate = self.get_rotation_matrix(landmarks, center_of_face, image, scale)
        warped = cv2.warpAffine(image, trotate, (width, height))
        
        
        return warped, self.average_angle
        
    def find_face_dim(self, landmarks, heigth, width):
        #find dimention of face in pixels as average of width and height
    
        xList = []
        yList = []
        for x,y in landmarks[0]:
            yList.append(y)
            xList.append(x)
    
    
        #y_max, y_min = int(np.max(yList) * heigth), int(np.min(yList) * heigth)
        #x_max, x_min = int(np.max(xList) * width) , int(np.min(xList) * width)
    
        y_max, y_min = int(np.max(yList) * heigth), int(np.min(yList) * heigth)
        x_max, x_min = int(np.max(xList) * width) , int(np.min(xList) * width)
    
    
    
    
    
        croped_points = (y_min,y_max, x_min,x_max)
    
        #dim = ((y_max - y_min) + (x_max - x_min))/2
        dim = ((y_max - y_min) + (x_max - x_min) +1200)
        return dim
    def get_rotation_matrix(self, landmarks, rotation_point, face_img, scale):
        # get rotation matrix as average angle between eyes points and mouse corner points
        #face points
        if self.angle == 0:
            left_eye_pt = landmarks[0][0]
            right_eye_pt = landmarks[0][1]
            left_mouth_pt = landmarks[0][3]
            right_mouth_pt =landmarks[0][4]
            #angles
            eye_angle = self.angle_between_2_pt(left_eye_pt, right_eye_pt)
            mouth_angle  = self.angle_between_2_pt(left_mouth_pt, right_mouth_pt)
            self.average_angle = (eye_angle + mouth_angle)/2 
        else:
            self.average_angle =  - self.angle
        
        
        
        #rotation
        M = cv2.getRotationMatrix2D((rotation_point[0], rotation_point[1]), self.average_angle, scale )
        return M
    def angle_between_2_pt(self,p1, p2):
        # to calculate the angle rad by two points   
        x1, y1 = p1
        x2, y2 = p2
        tan_angle = (y2 - y1) / (x2 - x1)
        return (np.degrees(np.arctan(tan_angle)))
    
    
    
    
    
    
    



    
    
    
    
    
