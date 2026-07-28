"""
@--22.08.2022--@
Author: github/farhadsh1992
INFO:
    Loads the pretrained StampOne decoder TFLite model to recover the
    hidden binary message from an encoded image, then error-corrects and
    decodes it back to the original text via BCH.
LAST_UPDATE:
"""
################################################################
# import onnxruntime
import numpy as np
import cv2
################################################################
import tensorflow as tf
import keras
# import torch
################################################################
# from Tools_Stega.Wavelet_transfer import wavelet_layer_all
# from Networks_StampOne_Lib.Wavelet_transfer_keras3 import  Wavelet_Layer_Keras3, Wavelet_Layer_Keras3_v2
# from Networks_StampOne_Lib.utils_preprocessing import Sobel_Egdes
################################################################
from stampone.FarhadCV.Tools import tcolors, bcolors
#################################################################################################
import bchlib

#################################################################################################
##            STampOne Deocder router                                                    ##
#################################################################################################
class TFLite_Decoder_Loader(keras.Model):
    

    def __init__(self, path_decoder:str="", device:str=''):
        super(TFLite_Decoder_Loader,self).__init__()
        
        path_decoder = ("./Tools_stega89_de1/tfLite_stampone89_deocder/"+
                        "StampOne_decoder_v89_intfl32_float16.tflite")
        # StampOne_decoder_v89_intfl32_float16.tflite
        # StampOne_decoder_v89_infl32_float32.tflite
        self.decoder_interpreter = tf.lite.Interpreter(model_path =path_decoder, 
                                                       experimental_preserve_all_tensors=True)

        self.secret_size = 256
        self.shape_message= [16, 16]
        self.pad_size = None
        self.image_size2 = 16
       

        self.image_size_de = 128
       

    

    def __call__(self, encoded_in:tf.uint8):

        #####################################################
        ##                  ##
        #####################################################
        encoded_in = tf.cast(encoded_in,        dtype="float32")
        encoded_in = tf.expand_dims(encoded_in, axis=0)
        encoded_in = tf.image.resize(encoded_in, (self.image_size_de,self.image_size_de))
        #####################################################
        ##                  ##
        #####################################################
        decoder_input_details = self.decoder_interpreter.get_input_details()
        decoder_output_details = self.decoder_interpreter.get_output_details()
        self.decoder_interpreter.allocate_tensors()
        
        self.decoder_interpreter.set_tensor(decoder_input_details[0]['index'], encoded_in)
        self.decoder_interpreter.invoke()
        message = self.decoder_interpreter.get_tensor(decoder_output_details[0]['index'])

        #####################################################
        ##                  ##
        #####################################################
        return message
    

    
#################################################################################################
######                                                                                    ######
#################################################################################################


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

###################################################################################################
########                                                           ########
###################################################################################################
def read_message(Message, BCH_BITS, BCH_POLYNOMIAL, bits, size, pad=0):


    
    if pad!=0:
        Message = Message[:, pad:-1*pad, pad:-1*pad, :]
    Message = tf.image.resize(Message, (size, size))
    Message =tf.image.rgb_to_grayscale(Message).numpy().astype("uint8")

    Message2 = np.reshape(Message, (1, size*size))
    Message2 = np.where((Message2 < 127), 1, Message2)
    Message2 = np.where((Message2 > 127), 0, Message2)
    decoded_msg = BCH_Reader(secret = Message2[0], BCH_BITS =BCH_BITS  , BCH_POLYNOMIAL=BCH_POLYNOMIAL, bits = bits)
    
    return decoded_msg


def BCH_Reader( secret, BCH_BITS, BCH_POLYNOMIAL, bits = 96):

  
    #print(secret)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    packet_binary = "".join([str(int(bit)) for bit in secret[:bits]])
    packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)
    
    
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
    bitflips = bch.decode_inplace(data, ecc)
  
    try:
        decoded_msg = data.decode("utf-8")
        print( tcolors.GREEN, "\n message:",decoded_msg, tcolors.ENDC)
        message_error = ""
            
    except Exception as e: 
        print( tcolors.RED, "Fail to decoder", tcolors.ENDC)
        decoded_msg = "None"
        message_error = f"{e}"
        
    return decoded_msg, message_error