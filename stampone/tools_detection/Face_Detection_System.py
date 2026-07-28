



"""
@--23.02.2023--@
Author: github/farhadsh1992
INFO:
    Face-detection wrapper (PRNet-based) that crops the face region for
    encoding and stitches the encoded face back into the source image.
LAST_UPDATE:  23.02.2023
    -
"""

import tensorflow as tf

import numpy as np
from  stampone.FarhadCV.Tools import tcolors,bcolors




class Face_Detection_System_v23(tf.keras.models.Model):
    """
    INPUTS:
        Kind_Model: (LbpCasade, vertix, kpt, dlib, Mobilenetboxes, MobilenetLandmarks_small)
        (I shoud add Mobilenetboxes, MTCNN)

    -> vertix, kpt is from PRNet model:  https://github.com/yfeng95/PRNet
    -> LbpCasade: Opencv
    -> dlib: 
    -> Mobilenetboxes:
    -> MobilenetLandmarks_small
    -> MTCNN
    ---------------------------------------------------
    OUTPUTS:
        -> def crop_face : coped_face
        -> def match_face: Stick encoded face into image and produce encoded images. 
    ---------------------------------------------------
    INFO:
    
    REF:
        - PRNet: https://github.com/yfeng95/PRNet
        - LbpCasade: https://github.com/shamangary/SSR-Net
        - dlib: http://dlib.net/face_landmark_detection.py.html
        - Mobilenet (facelib): https://github.com/kutayyildiz/facelib
        - 

    """
    def __init__(self, Kind_Model:str="kpt"):

        self.Kind_Model = Kind_Model

        if self.Kind_Model.lower() == "vertix".lower():
            import stampone.tools_detection.Tools_Face_Detection_System.FaceDecPRnet as FD
            self.face_model = FD.face_detection_by_vertix()

        elif self.Kind_Model.lower() == "kpt".lower():
            import stampone.tools_detection.Tools_Face_Detection_System.FaceDecPRnet as FD
            self.face_model = FD.face_detection_by_kpt()

    def crop_face(self, image:tf.uint8, norm:bool=False)->tf.uint8:
        
        self.image = image
        
        self.face, self.points = self.face_model.call(image)
        
        if norm:
            self.face = tf.cast(tf.expand_dims(self.face, axis=0), dtype="float32")/255.0
        
        return self.face, self.points
    
    
    def stick_face(self, encoded_face:tf.uint8)->np.uint8:

        y_min,y_max, x_min,x_max = self.points
        self.encoded_image = np.array(self.image).copy()
        self.encoded_image[y_min:y_max, x_min:x_max,:] = np.array(encoded_face)
        self.encoded_image = self.encoded_image.copy()

        return self.encoded_image
    def crop_face_de(self, image:tf.uint8, norm:bool=False):
        face, points = self.crop_face(image=image, norm=norm)
        return face, points