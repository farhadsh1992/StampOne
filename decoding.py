"""
Date: @--02.07.2024--@
Author:  github/farhadsh1992
INFO:

"""




import tensorflow as tf
from  tools_nvidia.utils import Configure_GPU
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files, mkdirfile
import argparse
from tools_decoder.tools_TFLite_decoder import read_message
import cv2
import pandas as pd

##################################################################################
#####                 #####
##################################################################################
def paramters():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_model',   help = "", type = str, default = "StampOne_vesion89")
    parser.add_argument('--gpu_devices',    help="select which GPU to run a job on", 
                        type=str,  default="0")
    parser.add_argument('--devices_number', help="select which GPU to run a job on", 
                        type=list, default=[0, 0, -1])
    parser.add_argument('--detector', help="detect the encoded part", 
                        choices=['None', 'FaceDetection', 'ObjectDetection', 
                                 'PinkBorder', 'QRCode'],
                        type=str, default='FaceDetection')
    parser.add_argument('--encoded_images', help="path file of encoded images ", 
                        type=str, default='./results/encoded_images/')
    parser.add_argument('--save_dir', help="path save", 
                        type=str, default='./results/')
    parser.add_argument('--save_decode_message', help="", 
                        type=bool, default=True)
    
    parser.add_argument('--BCH_BITS',    help = "",  
                        type = int,  default = 25)
    parser.add_argument('--BCH_POLYNOMIAL',    help="", 
                        type = int,  default = 487)
    parser.add_argument('--secret_size',    help = "", 
                        type = int,  default = 256)

    args = parser.parse_args()
    return args


##################################################################################
#####     Decoder Function            #####
##################################################################################
def decoder_router(args:argparse.ArgumentParser, devices:list):


    ## make files for saveing
    mkdirfile(args.save_dir)
    mkdirfile(args.save_dir + "Encoded_images/")
    mkdirfile(args.save_dir + "Encoded_parts/")
    ## read encoded images from folder
    name_images = read_files(args.encoded_images)
    ## for saveing message in csv
    decoded_message_list = []
    
    #######################################################
    ######        Load TFLite StampOne Router         ######
    #######################################################
    with tf.device(devices[0]):
        from tools_decoder.tools_TFLite_decoder import TFLite_Decoder_Loader
        StampOne_Decoder  = TFLite_Decoder_Loader()


    #######################################################
    ######        Load Detection Router         ######
    #######################################################
    ## detecting encoded part of  images
    from tools_detection.main import Detection_Models
    with tf.device(devices[0]): 
        Detect_Router = Detection_Models()
        name_decector = Detect_Router.choose_model(args.detector)

    #######################################################
    ######         decode images one by one        ######
    #######################################################
    for name in name_images:
        
        ## read images
        img = cv2.imread( args.encoded_images + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ## Crope encoded-part of the image
        with tf.device(devices[0]):
            try:
                croped_face, points = Detect_Router.crop(img)
                print(tcolors.GREEN, f"Apply {name_decector}", tcolors.ENDC)
            except Exception as e: 
                print(tcolors.GREEN, f"can not use {name_decector} - L89 - {e}", tcolors.ENDC)

        ## decode message from encoded images
        binery_message = StampOne_Decoder(croped_face)

        ## save decoded message
        if args.save_decode_message:
            binery_message2 = cv2.cvtColor(binery_message, cv2.COLOR_RGB2BGR)
            binery_message2 = cv2.imwrite(args.save_dir+name, binery_message2)

        ### read message by bchlibs
        message, message_error = read_message(Message        = binery_message, 
                                              BCH_BITS       = args.BCH_BITS , #25,  13, 7, 21, 44 
                                              BCH_POLYNOMIAL = args.BCH_BITS, #487#137#8219, 20023, 
                                              bits = args.secret_size ,#256, 
                                              size = 16, #16, 
                                              pad  = 0)
        
        decoded_message_list.append(message)
    ## save message in a csv file. 
    dict = {"name_images":name_images, "decoded_message":decoded_message_list}
    df = pd.DataFrame(dict)
    df.to_csv(args.save_dir+"decoded_message_results.csv")



if __name__ == '__main__':
    args = paramters()
    devices = Configure_GPU(args)
    decoder_router(args, devices)