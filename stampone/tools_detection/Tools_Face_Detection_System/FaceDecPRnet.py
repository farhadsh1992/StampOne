

"""
@--23.02.2023--@
Author: github/farhadsh1992
INFO:
    PRNet-based face crop by keypoints (kpt) or full vertex mesh (vertix),
    used by the 'FaceDetection' detector option.
LAST_UPDATE:
    - Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    - https://github.com/yfeng95/PRNet
    - https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf
"""

from skimage.transform import rescale, resize
#from api import PRN
import cv2
import numpy as np
import os
from stampone.FarhadCV.Tools import read_files, read_folders, estimater
from stampone.FarhadCV.Tools import tcolors
#os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU

from stampone.tools_detection.Tools_Face_Detection_System.api_tflite import PRN




class face_detection_by_kpt():
    def __init__(self, model_direction = ""):

        self.prn = PRN(is_dlib = True )

    def call(self, image):
    
    
        max_size = max(image.shape[0], image.shape[1])
   
        if max_size> 1000 and False:
            image = rescale(image, 1000./max_size)
            image = (image*255).astype(np.uint8)
            #print(tcolors.RED, "decresies image in kpt FaceDetection", tcolors.ENDC)
        
   
        pos = self.prn.process(image) # use dlib to detect face   
    
        kpt = self.prn.get_landmarks(pos)
    
        #
        first,  second, third = [],[],[]
        for ith,_ in enumerate(kpt):
            for jth, value in enumerate(kpt[ith]):
                if jth == 0:
                    first.append(value)
                if jth == 1:
                    second.append(value)
                if jth == 2:
                    third.append(value)    
    
        #
        img5 = np.array(image).copy()
        y_max, y_min = int(max(second)), int(min(second))
        x_max, x_min = int(max(first)), int(min(first))
        crop_img = img5[y_min:y_max, x_min:x_max]
        crop_img = crop_img.copy()
        croped_pints = [y_min, y_max, x_min,x_max]
        #plt.imshow(crop_img);plt.show()
        return crop_img, croped_pints




class face_detection_by_vertix():

    def __init__(self, model_direction = ""):
        self.prn = PRN(is_dlib = True )

    def call(self, image):
        max_size = max(image.shape[0], image.shape[1])
        if max_size> 1000 and False:
            image = rescale(image, 1000./max_size)
            image = (image*255).astype(np.uint8)
        pos = self.prn.process(image) # use dlib to detect face
    
        vertices = self.prn.get_vertices(pos)
        
        #
        first,  second, third = [],[],[]
        for ith,_ in enumerate(vertices):
            for jth, value in enumerate(vertices[ith]):
                if jth == 0:
                    first.append(value)
                if jth == 1:
                    second.append(value)
                if jth == 2:
                    third.append(value)    
    
        #
        img5 = image.copy()
        y_max, y_min = int(max(second)), int(min(second))
        x_max, x_min = int(max(first)), int(min(first))
        crop_img = img5[y_min:y_max, x_min:x_max]
        croped_pints = [y_min, y_max, x_min,x_max]
        #plt.imshow(crop_img);plt.show()
        return crop_img, croped_pints