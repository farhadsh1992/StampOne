"""
Date: @--02.07.2024--@
Author:  github/farhadsh1992
INFO:

"""


import tensorflow as tf
from  tools_nvidia.utils import Configure_GPU
from FarhadCV.Tools import tcolors, bcolors, estimator, read_files, mkdirfile
import cv2
import numpy as np
from tools_encoder.tools_TFLite_encoder import BCH_Generator

## 
def paramters():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_model',   help = "", type = str, default = "")
    parser.add_argument('--gpu_devices',    help="select which GPU to run a job on", 
                        type=str,  default="0")
    parser.add_argument('--devices_number', help="select which GPU to run a job on", 
                        type=list, default=[0, 0, -1])
    parser.add_argument('--detector', help="detect the  part for encoding", 
                        choices=['None', 'FaceDetection', 'ObjectDetection', 
                                 'PinkBorder', 'QRCode'],
                        type=str, default='')
    
    parser.add_argument('--original_images', help="path file of original images ", 
                        type=str, default='./results/encoded_images/')
    parser.add_argument('--save_dir', help="path save", 
                        type=str, default='./results/')
    
    parser.add_argument('--random_message', help="", 
                        type=bool, default=False)
    parser.add_argument('--message', help="", 
                        type=str, default="Visteam")
    
    parser.add_argument('--BCH_BITS',    help = "",  
                        type = int,  default = 25)
    parser.add_argument('--BCH_POLYNOMIAL',    help="", 
                        type = int,  default = 487)
    parser.add_argument('--secret_size',    help = "", 
                        type = int,  default = 256)


    args = parser.parse_args()
    return args


## Encoder Function
def encoder_router(args, devices):
    #######################################################
    ###
    mkdirfile(args.save_dir)
    mkdirfile(args.save_dir + "Encoded_images/")
    mkdirfile(args.save_dir + "Encoded_parts/")
    mkdirfile(args.save_dir + "2Dmessages/")
    #######################################################
    #######################################################s
    ## read images from folder
    name_images = read_files(args.encoded_images)
    #######################################################
    
        

    #######################################################
    ######        Load TFLite StampOne Router         ######
    #######################################################
    ## name model this file should write with big alphabets
    ## Load Encoder;
    with tf.device(devices[0]):
        from tools_encoder.tools_TFLite_encoder import TFLite_Encoder_Loader
        StampOne_Encoder = TFLite_Encoder_Loader()

    #######################################################
    ######        Load Detection Router         ######
    #######################################################
    from tools_detection.main import Detection_Models
    with tf.device(devices[0]): #gpu:1
        Detect_Router = Detection_Models()
        name_decector = Detect_Router.choose_model(args.detector)

    #########################################################################
    #####           read images and encode them            #####
    #########################################################################
    for name in name_images:
    
        img = cv2.imread( args.original_images + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      

        #########################################################################
        #####        make ready messages               #####
        #########################################################################
        if args.random_message:
            binery_message = (np.random.randint(2, size=(args.secret_size*1))).astype("uint8")
            binery_message = (binery_message).astype('float32')

            binery_message = np.reshape(binery_message*255, (16, 16, 1))
            binery_message = cv2.cvtColor(binery_message, cv2.COLOR_GRAY2RGB)
        else:
            binery_message = BCH_Generator(args.message,  
                                                BCH_POLYNOMIAL = int(args.BCH_POLYNOMIAL), 
                                                BCH_BITS = int(args.BCH_BITS), 
                                                number_zeros = 0, 
                                                sevensize = 1)
            binery_message = np.reshape(binery_message, (16, 16, 1))
            binery_message = cv2.copyMakeBorder(binery_message*255, top = 0, bottom = 0, 
                            left= 0, right = 0, 
                            borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

            binery_message = cv2.resize(binery_message, (16, 16), interpolation=cv2.INTER_NEAREST)
            binery_message = (binery_message/255.0).astype('float32')
            binery_message2 = cv2.cvtColor(binery_message, cv2.COLOR_GRAY2RGB)

            binery_message = tf.expand_dims(binery_message2, axis=0)
        #########################################################################
        ##### Crope encoded-part of the image           ####
        #########################################################################
        with tf.device(devices[0]):
            try:
                croped_face, points = Detect_Router.crop(img)
                print(tcolors.GREEN, f"Apply {name_decector}", tcolors.ENDC)
            except Exception as e: 
                print(tcolors.GREEN, f"can not use {name_decector} - L89 - {e}", tcolors.ENDC)

        #########################################################################
        #####        encode images                       #####
        #########################################################################
        with tf.device(devices[0]):
            encoded_face_image, output_size  = StampOne_Encoder(croped_face, binery_message)
            #########################################################################
            ## stick encoded part of encoded image:
            if name_decector == "PRNet-FaceDetection" or name_decector == "ObjectDetection":
                try:
                    encoded_image = Detect_Router.stick(encoded_face_image)
                except Exception as e: 
                    print(tcolors.RED, f"Can not  {name_decector} - {e}", tcolors.ENDC)

        
        
        #########################################################################
        #####           save Images                    #####
        #########################################################################
        ## save encoded images for showing and save by user
        encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.save_dir + f"/Encoded_images/{name}", encoded_image)

        encoded_face_image = cv2.cvtColor(encoded_face_image, cv2.COLOR_BGR2RGB)
        # encoded_face_image = cv2.resize(encoded_face_image, (size_img[1], size_img[0]))
        cv2.imwrite(args.save_dir + f"Encoded_parts/{name}", encoded_face_image)

        binery_message2 = cv2.cvtColor(binery_message2, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.save_dir + f"/2Dmessages/{name}", binery_message2)


if __name__ == '__main__':
    args = paramters()
    devices = Configure_GPU(args)
    encoder_router(args, devices)