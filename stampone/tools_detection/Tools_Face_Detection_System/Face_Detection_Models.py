

"""
@--23.02.2023--@
Author: github/farhadsh1992
INFO:
    Collection of alternative face-crop strategies (Haar cascade,
    MobileNet SSD boxes, MobileNet landmarks, mixes of the two) tried
    during development of the PRNet-based face detector.
LAST_UPDATE:  23.02.2023
    -
"""

from skimage.transform import rescale, resize
import cv2
import numpy as np
import os
from stampone.FarhadCV.Tools import read_files, read_folders, estimater, tcolors
from stampone.tools_detection.Tools_Face_Detection_System.Face_Detection_MobileNets import SSDFaceDetector, LandmarkDetector
#from mtcnn import MTCNN


#os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU



def face_detection_with_Haarcacade(image, 
                                   CasscadeModel = './DetectionLibs/Tools_Face_Detection_System/CasscadeModel/lbpcascade_frontalface_improved.xml'):
    face_cascade = cv2.CascadeClassifier(CasscadeModel)
    face_values = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h)  in face_values:
        x, y, w, h = x, y, w, h
        
    face = image[y:y+h,x:x+w]
    face = face.copy()
    return face, [y,y+h,x,x+w]



class Mobilenetboxes():
    def __init__(self, 
                 model_direction:str = "./DetectionLibs/Tools_Face_Detection_System/MobilenetModels/ssd_int8_cpu.tflite"):
      self.face_detect_router = SSDFaceDetector(model_direction)
    
    def call(self, image):
      height, width,_ = image.shape
     

       # image = np.cast(image, dtype="uint8")

      bboxes = self.face_detect_router.predict(image)

   
      x_min = int(bboxes[0][0][1] * width)
      x_max = int(bboxes[0][1][1] * width)
    
      y_min = int(bboxes[0][0][0] * height)
      y_max = int(bboxes[0][1][0] * height)            
      croped_points =  (y_min,y_max, x_min,x_max)
      crop_img = np.array(image[y_min:y_max, x_min:x_max]).copy()
    
      return crop_img , croped_points

def MobilenetboxesHalf(image, 
                       model_direction = "./DetectionLibs/Tools_Face_Detection_System/MobilenetModels/ssd_int8_cpu.tflite"):

    
    height, width,_ = image.shape
    facedec = SSDFaceDetector(model_direction)
    bboxes = facedec.predict(image)
   
    x_min = int((bboxes[0][0][1] * width) +( bboxes[0][0][1] * width *(2.0/10.0)) )
    x_max = int(bboxes[0][1][1] * width *(9.0/10.0))
    
    y_min = int((bboxes[0][0][0] * height) + (bboxes[0][0][0] * height*(300.0/100.0)) )
    y_max = int(bboxes[0][1][0] * height *(8.0/10.0))            
    croped_points =  (y_min,y_max, x_min,x_max)
    crop_img = image[y_min:y_max, x_min:x_max].copy()
    
    return crop_img , croped_points



def MobilenetLandmarks_small(image, model_direction):
    
    #heigth, width, _ = image.shape
    width, heigth, _ = image.shape
    
    LD =  LandmarkDetector(model_direction)
    landmarks = LD.predict(image)
    
    
    xList = []
    yList = []
    for x,y in landmarks[0]:
        yList.append(y)
        xList.append(x)
    
    
    x_max, x_min = int(np.max(yList) * heigth), int(np.min(yList) * heigth)
    y_max, y_min  = int(np.max(xList) * width) , int(np.min(xList) * width)
    
    #y_max, y_min = int(np.max(yList) * heigth), int(np.min(yList) * heigth)
    #x_max, x_min = int(np.max(xList) * width) , int(np.min(xList) * width)
    
    
    
    
    
    croped_points = (y_min,y_max, x_min,x_max)
    crop_img = image[y_min:y_max, x_min:x_max].copy()
    #(y_min,y_max, x_min,x_max)
    return crop_img , croped_points

def Landmark_MTCNN(image, model_direction=""):
    detector = MTCNN()
    result = detector.detect_faces(image)
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']
    
    return crop_img , croped_points

def MobilenetMix(image, model_direction = ""):
    

    model_direction_1 = "./DetectionLibs/Tools_Face_Detection_System/MobilenetModels/ssd_int8_cpu.tflite"
    model_direction_2 = "./DetectionLibs/Tools_Face_Detection_System/MobilenetModels/mobilenet_fp32_cpu.tflite"
    
    crop_img_1 , croped_points_1 =  Mobilenetboxes(image, model_direction_1)
    crop_img_2 , croped_points_2 =  MobilenetLandmarks_small(crop_img_1, model_direction_2)
    
    return crop_img_1 , crop_img_2, croped_points_1, croped_points_2



def LpbcacadeMix(image, model_direction = ""):
    

    model_direction_1 = "./DetectionLibs/Tools_Face_Detection_System/MobilenetModels/ssd_int8_cpu.tflite"
    model_direction_2 = "./DetectionLibs/Tools_Face_Detection_System/CasscadeModel/lbpcascade_frontalface_improved.xml"
    
    crop_img_1 , croped_points_1 =  Mobilenetboxes(image, model_direction_1)
    crop_img_2 , croped_points_2 =  face_detection_with_Haarcacade(crop_img_1, model_direction_2)
    
    return crop_img_1 , crop_img_2, croped_points_1, croped_points_2