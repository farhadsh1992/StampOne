"""
@--30.08.2022--@
Author: github/farhadsh1992
INFO:
    GPU/device configuration and environment-diagnostics helpers used
    before loading the StampOne encoder/decoder models.
    -ref:
        - https://eladrich.github.io/pixel2style2pixel/
    	- https://github.com/eladrich/pixel2style2pixel

LAST_UPDATE:
"""


import os
import sys
import numpy as np
import pkg_resources

import tensorflow as tf

from pprint import pprint
from stampone.FarhadCV.Tools import tcolors, bcolors
# from utils import common



###################################################################################################
#########                                                                       #########
###################################################################################################
def CHECK_PYTHON_SETTING():
    print(tcolors.RED)
    pprint({
        'PATH': os.environ['PATH'].split(os.pathsep),
        'PYTHONPATH': get_pythonpath(),
        'sys.path': sys.path,
        'sys.executable': sys.executable,
        'sys.prefix': sys.prefix,
        'sys.version_info': sys.version_info,
        'pkg_resources.working_set': list(pkg_resources.working_set),
    })
    print(tcolors.ENDC)

###################################################################################################
#########                                                                       #########
###################################################################################################
def get_pythonpath():
    try:
        return os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        return None

###################################################################################################
#########                                                                       #########
###################################################################################################
def Configure_GPU(args=None):

    # import torch
    # torch.cuda.set_device(0)
    # # On device 0
    # with torch.cuda.device(1):
    #     print("Inside device is 1")  
    # Pin GPU to be used to process local rank (one GPU per process)
    # logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(bcolors.WHITE+tcolors.BLUE, "DEVICES: ", gpus,  tcolors.ENDC) 
    devices = []
    for i, gpu in enumerate(gpus):
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(gpu, True)
        devices.append(f'/gpu:{i}')
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(bcolors.WHITE+tcolors.BLUE,len(gpus), "Physical_GPUs", len(logical_gpus), "logical GPUS", tcolors.ENDC)
    
    if gpus == []:
        print(bcolors.WHITE+tcolors.BLUE, "CAN NOT LOAD ANY GPU DEVICES - CPU",  tcolors.ENDC) 
        devices= ['/gpu:0', '/gpu:0', '/cpu:0']
    else:
        print(bcolors.WHITE+tcolors.BLUE, "DEVICES: ", gpus,  tcolors.ENDC)  
    if len(devices) < 2:
        devices.append('/gpu:0')
    devices.append('/cpu:0')
    return devices

    


###################################################################################################
#########                                                                       #########
###################################################################################################



