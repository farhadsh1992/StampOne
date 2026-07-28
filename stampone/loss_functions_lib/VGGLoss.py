"""
@nov 2022@
author: https://github.com/farhadsh1992

INFO: VGG16/19 perceptual loss comparing cover and encoded image feature
maps (MSE on VGG activations) instead of raw pixel error.
"""

import tensorflow as tf
import numpy as np
from stampone.FarhadCV.Tools import tcolors
import os 

import time

import warnings
tf.get_logger().setLevel("DEBUG")
warnings.filterwarnings('ignore')














class VGGLoss(tf.keras.layers.Layer):
    """
    INPUTS: images(224×224×3), encoded-images(224×224×3)
    -----------------------------------------------------------------------
    OUPUTS:
        - Outputs of VGG-features extractor: 7 x 7 x 512
    -----------------------------------------------------------------------
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    
    https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0
    https://www.kaggle.com/code/ryanmarfty/extracting-feature-by-vgg19
    """
    def __init__(self, vgg_version: str="vgg16"):
        super(VGGLoss, self).__init__()


        if vgg_version=="vgg16":
            self.VGGModel = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
        elif vgg_version=="vgg19":
            self.VGGModel = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

        self.mse_loss = tf.keras.losses.MeanSquaredError()


    def call(self, images, encoded_images):
        vgg_on_cov = self.VGGModel(images)
        vgg_on_enc = self.VGGModel(encoded_images)
        g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
        
        return g_loss_enc