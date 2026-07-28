
"""
TFLite SSD face-box detector and facial-landmark detector, used by the
PRNet face pipeline (api_tflite.py) to locate faces without dlib.
"""

# import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from  stampone.FarhadCV.Tools import tcolors
from tensorflow import lite as tflite
import tflite_runtime.interpreter as tflite



class SSDFaceDetector():

    def __init__(self, tflite_model, input_img_channel='rgb', resize=None, tpu=False):
        
        self.face_detection_inference = tflite.Interpreter(tflite_model)
        self.face_detection_inference.allocate_tensors()
        self.index_input = self.face_detection_inference.get_input_details()[0]['index']
        self.index_boxes = self.face_detection_inference.get_output_details()[0]['index']
        self.index_probs = self.face_detection_inference.get_output_details()[2]['index']
        assert input_img_channel in ['bgr', 'rgb'], 'Incorrect input_img_channel'
        self.input_img_channel = input_img_channel

    def _preprocess(self, img):
        if self.input_img_channel == 'bgr':
            img = img[...,::-1] # bgr to rgb conversion
        img = cv2.resize(img, (320, 320))
        img = np.expand_dims(img, 0)
        return img

    @staticmethod
    def _extract_boxes(boxes, probs):
        threshold = 1e-1
        mask = np.less(probs, 1.)
        probs = probs[mask]
        mask = np.greater(probs, threshold)
        num_boxes = np.count_nonzero(mask)
        output = boxes[0][:num_boxes]
        return output

    def predict(self, img):
        """Detect faces inside an image.

        Parameters
        ----------
        img : ndarray
            3d array with shape (height,width,3) with dtype `uint8`.
        """
        original_shape = img.shape
        img = self.squarize_img(img)
        squared_shape = img.shape
        img = self._preprocess(img) #ok
            

           
        self.face_detection_inference.set_tensor(self.index_input, img)
        self.face_detection_inference.invoke()
        pred_boxes = self.face_detection_inference.get_tensor(self.index_boxes) #(1, 50, 4)
        pred_probs = self.face_detection_inference.get_tensor(self.index_probs)
        bboxes1 = self._extract_boxes(pred_boxes, pred_probs)
        bboxes1 = bboxes1.reshape(-1, 2, 2)
        fix = bboxes1[:,1,1] - bboxes1[:,0,1]
        fix = fix / 6.
        fix = np.expand_dims(fix, -1)
        fix = fix * [0., 1, 0, -1]
        fix = fix.reshape(-1, 2, 2)
        bboxes1 = bboxes1 + fix
        bboxes1 = self.normalize_bboxes(bboxes1, original_shape, squared_shape)
        return bboxes1

    @staticmethod
    def squarize_img(img):
        # batch, height, width = img.shape[:3]
        height, width = img.shape[:2]

        max_dim = max(height, width)
        # new_shape = (batch, max_dim, max_dim, 3)
        new_shape = (max_dim, max_dim, 3)

        # squarized = np.zeros(new_shape, dtype=np.float32)
        squarized = np.zeros(new_shape, dtype=np.uint8)

        squarized[:height,:width] += img
        # squarized[:, :height,:width] += img

        return squarized

    @staticmethod
    def normalize_bboxes(bboxes, original_shape, squared_shape):
        change_ratio = np.array(squared_shape[:2], dtype=bboxes.dtype) / original_shape[:2]
        bboxes = bboxes * change_ratio
        # bboxes y_min, x_min, y_max, x_max
        return bboxes
    
    
    
    
    
class LandmarkDetector:
    def __init__(self, tflite_model, input_img_channel='rgb'):
        #name_model, inference, tpu = name.split('_')
        
        #assert inference in ['fp32', 'int8'], 'Available inferences are: fp32, int8'
        

        self.landmark_detection_inference  = tflite.Interpreter(model_path = tflite_model)
        
        
        self.landmark_detection_inference.allocate_tensors()
        self.input_index = self.landmark_detection_inference.get_input_details()[0]['index']
        self.output_index = self.landmark_detection_inference.get_output_details()[0]['index']

        assert input_img_channel in ['bgr', 'rgb'], 'Incorrect input_img_channel'
        self.input_img_channel = input_img_channel

    def predict(self, img):
        img = cv2.resize(img, (96, 96))
        if self.input_img_channel == 'bgr':
            img = img[...,::-1]
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.
        self.landmark_detection_inference.set_tensor(self.input_index, img)
        self.landmark_detection_inference.invoke()
        landmarks = self.landmark_detection_inference.get_tensor(self.output_index)[0]
        landmarks = landmarks.reshape(-1, 5, 2)
        return landmarks