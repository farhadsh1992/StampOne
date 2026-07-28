





"""
'ObjectDetection' detector: runs a TFLite SSD-MobileNet model to find and
crop the largest detected object as the region to encode/decode.
"""

import tensorflow as tf
import numpy as np
from stampone.tools_detection.Tools_ObjectDetection.seg_tflite import get_output_dict, create_category_index
import tflite_runtime.interpreter as tflite
from stampone.tools_detection.Tools_ObjectDetection.utiles_OD import Copper_object
import cv2
from stampone.FarhadCV.Tools import tcolors, bcolors

class YoloV5_Object_Detections():
    def __init__(self, path:str=""):
        model_file = "./DetectionLibs/Tools_ObjectDetection/ModelTFLITE/ssd_mobilenet_v1_1_metadata_1.tflite"
        self.nms = True # bool -  To perform non-maximum suppression or not. The default is True.
        self.iou_thresh = 0.5 # Intersection Over Union Threshold. The default is 0.5.
        self.score_thresh = 0.6 # score above predicted class is accepted. The default is 0.6.
        
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        
    def crop_face(self,image:np.array):
        self.image = image.copy()
        # image = tf.cast(image, dtype="uint8")
        image = cv2.resize(image, (300,300))
        image = np.expand_dims(image, axis=0)
        

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
    
        de_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        det_classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        det_scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_det = self.interpreter.get_tensor(self.output_details[3]['index'])[0]


        category_index = create_category_index(label_path='./DetectionLibs/Tools_ObjectDetection/coco_ssd_mobilenet/labelmap.txt')
        output_dict =  get_output_dict(image, self.interpreter, self.output_details, self.nms, self.iou_thresh, self.score_thresh)


        croped_object, self.points = Copper_object(
                                        self.image,
                                        output_dict['detection_boxes'],
                                        output_dict['detection_classes'],
                                        output_dict['detection_scores'],
                                        category_index,
                                        use_normalized_coordinates=True,
                                        min_score_thresh=0.6,
                                        line_thickness=3)
        #points = y_min,y_max, x_min,x_max
        return croped_object, self.points
    def stick_face(self, encoded_face:np.uint8)->np.uint8:
        (left, right, top, bottom) = self.points
        self.encoded_image = np.array(self.image).copy()
        print(tcolors.BLUE," self.points:",  self.points,tcolors.ENDC)
        self.encoded_image[top:bottom,left:right,:] = np.array(encoded_face)
        self.encoded_image5 = self.encoded_image.copy()
        return self.encoded_image5