"""
TFLite version of PRNet's PosPrediction: runs the position-map regression
network via a TFLite interpreter instead of a TF1 session.
"""

# import tensorflow as tf
import numpy as np
from stampone.FarhadCV.Tools import tcolors
from tensorflow import lite as tflite
import tflite_runtime.interpreter as tflite


class PosPrediction():
    def __init__(self, resolution_inp = 256, resolution_op = 256): 
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp*1.1

        

    def restore(self, model_path):   
        #LiteModel = "./DetectionLibs/PRnet_TFLite_models/PRnet_pose_model_f32.tflite"
        #LiteModel = "./DetectionLibs/PRnet_TFLite_models/PRnet_pose_model_f16.tflite"
        LiteModel = "./DetectionLibs/Tools_Face_Detection_System/PRnet_TFLite_models/PRnet_pose_model_int8.tflite"
        self.interpreter = tflite.Interpreter(model_path = LiteModel)
        self.interpreter.allocate_tensors()
 
    def predict(self, image):
        #print(image.shape)
        #print(image.dtype)
        image = image.astype(np.float32)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
    
        self.interpreter.set_tensor(input_details[0]['index'], image[np.newaxis, :,:,:])
        self.interpreter.invoke()
        pos = self.interpreter.get_tensor(output_details[0]['index'])
        pos = np.squeeze(pos)
        return pos*self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: images})
        return pos*self.MaxPos