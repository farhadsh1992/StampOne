
"""
@--22.12.2023--@
Author: github/farhadsh1992
INFO:
    Shift-Tolerant LPIPS (ST-LPIPS) TFLite metric: a perceptual-similarity
    score that is more robust to small spatial shifts, useful for judging
    encoded images after print/scan misalignment.
    -ref:

LAST_UPDATE:
"""

import os
import tensorflow as tf
import keras
tf.get_logger().setLevel("DEBUG")


#######################################################################
def im2tensor(path:str="", image:tf.Tensor=""):
    if path != "":
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image)
      
        image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (416 , 416))
    # image = tf.expand_dims(image, axis=0)
    image = tf.transpose(image, (0,3,1,2))
    # image =  tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,3,1,2)))(image[0])
    return image
#########################################################################################
######                          ######
#########################################################################################

@keras.utils.register_keras_serializable(package="lpips_metric_lite")
class lpips_metric_lite(tf.keras.layers.Layer):
    def __init__(self, tflite_model_path: str="./loss_functions_lib/lpips_model/ShiftTolerant_LPIPS_TFLite.tflite"):
        super(lpips_metric_lite, self).__init__()
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # get inputs and output shapes
        input_shape1 = self.input_details[0]['shape']
        input_shape2 = self.input_details[1]['shape']
        output_shape = self.output_details[0]['shape']

        
        
    # @tf.function(jit_compile=True)
    def __call__(self, img0: tf.Tensor, img1: tf.Tensor)-> float:
        self.interpreter.set_tensor(self.input_details[0]['index'], img0)
        self.interpreter.set_tensor(self.input_details[1]['index'], img1)

    
        self.interpreter.invoke()

        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0][0][0]
       