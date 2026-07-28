"""
@--19.08.2022--@
Author: github/farhadsh1992
INFO:
    StegaStamp-style reveal/full losses (secret + cover reconstruction
    error) plus a YUV-weighted, edge-falloff image loss and an optional
    E-LPIPS perceptual-distance wrapper.
LAST_UPDATE:
"""


#import elpips
import tensorflow as tf
import numpy as np
import warnings

tf.get_logger().setLevel("DEBUG")
warnings.filterwarnings('ignore')






# Variable used to weight the losses of the secret and cover images (See paper for more details)
beta = 1.0
    
# Loss for reveal network
def rev_loss(s_true, s_pred):
    # Loss for reveal network is: beta * |S-S'|
    result = tf.math.reduce_sum(tf.square(s_true - s_pred))
    return result

# Loss for the full model, used for preparation and hidding networks
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[...,0:3], y_true[...,3:6]
    s_pred, c_pred = y_pred[...,0:3], y_pred[...,3:6]
    
    s_loss = rev_loss(s_true, s_pred)
    #c_loss = tf.keras.losses.MSE(c_true, c_pred)
    c_loss = tf.math.reduce_sum(tf.square(c_true -  c_pred))
    result = s_loss + c_loss
    return result


def Encoder_lossOS(original_image, encoded_image):
    pass

def Decoder_lossOS(original_image, encoded_image):
    pass





def Loss_OP(encoded_image, image_input, l2_edge_gain, yuv_scales_pl):

    size = (int(image_input.shape[1]),int(image_input.shape[2]))
    gain = 10
    yuv_scales = yuv_scales_pl

    falloff_speed = 4 # Cos dropoff that reaches 0 at distance 1/x into image
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0]/falloff_speed)):
        falloff_im[-i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
        falloff_im[i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
    for j in range(int(falloff_im.shape[1]/falloff_speed)):
        falloff_im[:,-j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
        falloff_im[:,j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
    falloff_im = 1-falloff_im
    falloff_im = tf.convert_to_tensor(falloff_im, dtype=tf.float32)
    falloff_im *= l2_edge_gain


    encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
    image_input_yuv = tf.image.rgb_to_yuv(image_input)
    im_diff = encoded_image_yuv-image_input_yuv
    im_diff += im_diff * tf.expand_dims(falloff_im, axis=[-1])
    yuv_loss_op = tf.reduce_mean(tf.square(im_diff), axis=[0,1,2])
    image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)
    return image_loss_op




class E_LPIPS():
    def __init__(self, batch_size):
        self.metric = elpips.Metric(elpips.elpips_vgg(batch_size=batch_size))
    def get_inputs(self, original_images):
        self.original_images = original_images
    def get_labels(self, encoded_images):
        self.encoded_images = encoded_images
    def compute(self):
        tf_distance = self.metric.forward(self.original_images, self.encoded_images)
        return tf_distance