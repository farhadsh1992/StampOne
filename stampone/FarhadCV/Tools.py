"""
General-purpose filesystem, terminal-color, and progress-logging helpers
shared across StampOne's encoder/decoder/detection modules.
"""

from os import listdir
from os.path import isfile, join
import  os

import datetime
import pendulum
import psutil
import sys
import numpy as np
import cv2
import shutil
import warnings
#tf.get_logger().setLevel("INFO")
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import shutil
def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def estimator(num,len_t, description):
    """
    info:
    --------------------------------------
    input:
    --------------------------------------
    output:
    
    """
    num+=1
    run = (tcolors.GREEN+"["+str(num)+'/'+str(len_t)+"]["+str(description)+']'+tcolors.ENDC)
        
    sys.stdout.write('\r'+ run)
    

def read_files(mypath):
    """
    info:
    --------------------------------------
    input:
    --------------------------------------
    output:
    
    """
    try:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, name))]
    except:
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
    return onlyfiles

def read_folders(mypath):
    """
    info:
    --------------------------------------
    input:
    --------------------------------------
    output:
    
    """
    try:
        onlyfolders = [ name for name in listdir(mypath) if isdir(join(mypath, name)) ]
    except:
        from os import listdir
        from os.path import isdir, join
        onlyfolders = [ name for name in listdir(mypath) if isdir(join(mypath, name)) ]
        
    return onlyfolders
    
def Covert_Array_to_image(input_array):
    open_cv_image = np.array(input_array) 
    # Convert RGB to BGR 
    input_array = open_cv_image[:, :, ::-1].copy() 
    output_img = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
    return output_img
    
class tcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    Orange = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\035[4m'
    WHITE =  '\033[37m'
 
# bachground color
class bcolors: 
    #HEADER = '\033[95m'
    BLUE = '\033[44m'
    GREEN = '\033[42m'
    WHITE = '\033[47m'
    RED = '\033[41m'
    BLACK = '\033[40m'
    BOLD = '\033[1m'
    PURPLE = '\035[45m'
    ENDC = '\033[0m'
    BOX_BlUE = '\033[44m'+'                                         '+'\033[0m'    
    

def mkdirfile(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def copy_files(source, distincation):
    # source = 'source_file.txt'
    # distincation = 'destination_file.txt'
    if  os.path.exists(distincation):
        os.remove(distincation)  
    shutil.copy(source, distincation)

def copy_folder(source, distincation):
    # source = 'source_file.txt'
    # distincation = 'destination_file.txt'
    if  os.path.exists(distincation):
        os.remove(distincation + "/*.py")
        os.rmdir(distincation)  
    shutil.copytree(source, distincation) 