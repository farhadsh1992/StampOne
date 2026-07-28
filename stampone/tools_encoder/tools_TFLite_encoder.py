"""
@--22.08.2022--@
Author: github/farhadsh1992
INFO:
    Loads the pretrained StampOne encoder TFLite model and blends its
    residual output into the cover image to produce the encoded image;
    also builds the BCH-protected binary message that gets embedded.
LAST_UPDATE:
"""
################################################################
# import onnxruntime
import numpy as np
import cv2
################################################################
import tensorflow as tf
from tensorflow import lite as tflite
import tflite_runtime.interpreter as tflite

import keras
# import torch
################################################################
# from Tools_Stega.Wavelet_transfer import wavelet_layer_all
# from Networks_StampOne_Lib.Wavelet_transfer_keras3 import  Wavelet_Layer_Keras3, Wavelet_Layer_Keras3_v2
# from Networks_StampOne_Lib.utils_preprocessing import Sobel_Egdes
################################################################
from stampone.FarhadCV.Tools import tcolors, bcolors
import bchlib
#################################################################################################
##                                                                                           ##
#################################################################################################
class TFLite_Encoder_Loader(keras.Model):
    def __init__(self, path_encoder=""):
        super(TFLite_Encoder_Loader, self).__init__()

        # /tfLite_models_inputsf32/StampOne_Encoder_v89_4_inputf32_float16.tflite
        # /tfLite_models_inputsf32/StampOne_Encoder_v89_inputsf32_float32.tflite
        path_encoder = (f"./Tools_Stega89_en/tfLite_models_inputsf32/"+
                      "StampOne_Encoder_v89_4_inputf32_float16.tflite")
        self.encoder_interpreter = tflite.Interpreter(model_path =path_encoder, 
                                                       experimental_preserve_all_tensors=True)
        
        self.encoder_input_details = self.encoder_interpreter.get_input_details()
        self.encoder_output_details = self.encoder_interpreter.get_output_details()

        # self.batch_size = 1
        # self.secret_size = 256
        # self.shape_message= [16, 16]
        # self.pad_size = None
        # self.image_size2 = 16
        # self.image_size = 256

        
    def __call__(self, image:tf.uint8, message:tf.uint8):

        
       
        #####################################################
        ##                  ##
        #####################################################
        ##
        image2 = tf.cast(image, dtype="float32")
        message2 = tf.cast(message, dtype="float32")
        image2 = tf.expand_dims(image2, axis=0)
        image2 = tf.image.resize(image2, (256,256))
        # tf.expand_dims(, axis=0)

        # print(tcolors.RED,"image2",image2,tcolors.ENDC)
        # print(tcolors.RED,"message2",message2,tcolors.ENDC)


        
        # encoder_input_details = self.encoder_interpreter.get_input_details()
        # encoder_output_details = self.encoder_interpreter.get_output_details()
        self.encoder_interpreter.allocate_tensors()
        self.encoder_interpreter.set_tensor(self.encoder_input_details[1]['index'], message2)
        self.encoder_interpreter.set_tensor(self.encoder_input_details[0]['index'], image2)
        self.encoder_interpreter.invoke()
        residual = self.encoder_interpreter.get_tensor(self.encoder_output_details[0]['index'])
        # print()
        #####################################################
        ##                  ##
        #####################################################
        # print(tcolors.RED,"residual",residual,tcolors.ENDC)
        encoded_image, output_size = maker_ready_encoded_image(image, 
                                                               residual, 
                                                               resduial_coefficient=1)
        encoded_image = np.array(encoded_image[0], dtype='uint8')

        # print(tcolors.RED,"encoded_image: ", encoded_image,tcolors.ENDC)
        blend_image = cv2.addWeighted(encoded_image,0.9, image, 0.1,0)

        return blend_image, output_size
    
   
#################################################################################################
##                                                                                           ##
#################################################################################################
def maker_ready_encoded_image(image, residual, resduial_coefficient=1):
                
    output_size = (image.shape[0], image.shape[1] )
    image2 = tf.cast(image, dtype="float32")/255.0
    ii2 = tf.image.resize(image2, output_size)
    residual2 = tf.image.resize(residual, output_size)


    encoded_gen_output = (resduial_coefficient*residual2) + (1*ii2) 
    encoded_gen_output = keras.ops.clip(encoded_gen_output, 0, 1)
    encoded_gen_output = keras.ops.cast((encoded_gen_output * 255), dtype="uint8")
    # encoded_blend = tfa.image.blend(encoded_gen_output, image, 0.2).numpy().astype("uint8")
            
    return encoded_gen_output, output_size

###################################################################################################
########                                                           ########
###################################################################################################
def normalize_fixed(x, current_range, normed_range):
    # current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    current_min, current_max = current_range[0], current_range[1]

    # normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    normed_min, normed_max = normed_range[0], normed_range[1]

    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed
###################################################################################################  


def BCH_Generator(secret, BCH_POLYNOMIAL = 137, BCH_BITS = 7, number_zeros = 4, sevensize = 1):
    
    #print(tcolors.RED, secret, tcolors.ENDC)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    data = bytearray(secret + ' '*(sevensize-len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0 for i in range(number_zeros)])
    return secret