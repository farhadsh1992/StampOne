"""
Router that picks and drives whichever detector locates the encoded
region of an image (face detection, object detection, colored border, or
QR pattern) so encoding.py/decoding.py can crop/stick that region.
"""

# from core.models import selector1
from stampone.tools_detection.Face_Detection_System import Face_Detection_System_v23
from stampone.tools_detection.Border_Detection_Sys import add_purple_border
# from DetectionLibs.Yolov5 import YoloV5_Object_Detections


import tensorflow as tf
from stampone.FarhadCV.Tools import tcolors, bcolors

class Detection_Models():
    def __init__(self,):

        ## use select the model
        ## 0 = face-detection,# 1=object-detcetion,# 2=pink-border
        pass

    def choose_model(self, num_model):
        # num_model  = selector1(request)
        print(tcolors.GREEN,"num_model",num_model, tcolors.ENDC)
        self.num_model = num_model
        if num_model == "FaceDetection" or num_model==None:
            name_decector = "PRNet-FaceDetection"
            self.detector_router = Face_Detection_System_v23(Kind_Model="kpt")
        elif num_model == "ObjectDetection":
            name_decector = "Yolo-ObjectDetection"
            from stampone.tools_detection.Yolov5 import YoloV5_Object_Detections
            self.detector_router = YoloV5_Object_Detections()
        elif num_model == "PinkBorder":
            name_decector = "OpenCV-BorderDetection"
            self.detector_router = add_purple_border()
        elif num_model == "QRCode":
            name_decector = "QRCode-Pattern"
        else:
            raise NotImplementedError()
        return name_decector
    def crop(self, encoded_img:tf.Tensor):
        
        croped_image = self.detector_router.crop_face(encoded_img)
       
        return croped_image
    
    def stick(self, encoded_img:tf.Tensor, points:tuple=None):
        encoded_image = self.detector_router.stick_face(encoded_img)
        return encoded_image
    
    def crop_de(self, encoded_img:tf.Tensor):
        
        croped_image = self.detector_router.crop_face_de(encoded_img)
       
        return croped_image