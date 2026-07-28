









"""
RGB-uv color histogram loss layer: builds a differentiable log-chroma
histogram for an image and compares it between cover and encoded images
to keep their color distributions close (HistoGAN-style color loss).
"""

import tensorflow as tf
import keras
import numpy as np
import warnings
tf.get_logger().setLevel("DEBUG")
warnings.filterwarnings('ignore')





# ----------------------------------------------------------------------------------------------------------------------
# +++++++++++++++++<Histogram Loss>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ++++++++++++++++++++++++  

@keras.utils.register_keras_serializable(package="RGBuvHistBlock_loss") 
class RGBuvHistBlock_forRecoverNet(keras.models.Model):

    def __init__(self, h:int=64, insz:int=150, 
                 resizing:str='interpolation', 
                 method:str='inverse-quadratic', sigma:float=0.02, 
                 intensity_scale:bool=True, **kwargs):
        super(RGBuvHistBlock_forRecoverNet, self).__init__(**kwargs)
        
        self.h = h
        
        #  maximum size of the input image
        self.insz = insz

        
        # resizing method if applicable: 'interpolation' or 'sampling'
        self.resizing = resizing 
        
        
        # the method used to count the number of pixels for each bin in the  histogram feature. 
        # Options are: 'thresholding', 'RBF' (radial basis function), or 'inverse-quadratic'
        self.method = method 
       
        #  if the method value is 'RBF' or 'inverse-quadratic', then this is  the sigma parameter of the kernel function.
        self.intensity_scale = intensity_scale 
        self.EPS = 1e-6
        
        
        if self.method == 'thresholding':
            self.eps = 6.0 / h
        else:
            self.sigma = sigma
            
    def call(self, x):
        
        channel = 3
        x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
        
        # resize network if the picture is biger than maxsize
        if x.shape[1] > self.insz or x.shape[2] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = tf.image.resize(x, size=(self.insz, self.insz), method='bilinear')
            elif self.resizing == 'sampling':
                x_sampled = tf.image.resize(x, size=(self.insz, self.insz),  method='nearest')
                
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
                
        else:
            x_sampled = x
            
        batch = x_sampled.shape[0]  # size of mini-batch
        
        # if image has more 3 channel color
        if x_sampled.shape[3] > 3:
            x_sampled = x_sampled[:, :, :, :3]
            
        
            
        # sepret batch images form each other 
        #Xs_spereted_bacth = tf.split(x_sampled, num_or_size_splits = batch, axis=0)
        Xs_spereted_bacth = tf.unstack(x_sampled, axis=0)
      
     
        hists = np.zeros((batch, self.h, self.h, 3), dtype='float32')
        
        for lth in range(batch):
            
            #--------------------------------->>>
            I = tf.reshape(Xs_spereted_bacth[lth], (-1, 3))
            #I  = tf.transpose(I)
   
            II = tf.pow(I, 2)
            #--------------------------------->>>
            
            if self.intensity_scale:
                Iy = tf.expand_dims(tf.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + self.EPS),  axis=1)
            else:
                Iy = 1
            #--------------------------------->>>
            for ich in range(channel):
                
                #--------------------------------->>>
                if ich == 0:
                    ich_1 = 1
                    ich_2 = 2
                if ich == 1:
                    ich_1 = 0
                    ich_2 = 2
                if ich == 2:
                    ich_1 = 0
                    ich_2 = 1
                #--------------------------------->>>
                
                #--------------------------------->>>
       
                Iu =  tf.expand_dims( tf.math.log(I[:, ich] + self.EPS) - tf.math.log(I[:, ich_1] + self.EPS),  axis=1)
                Iv =  tf.expand_dims( tf.math.log(I[:, ich] + self.EPS) - tf.math.log(I[:, ich_2] + self.EPS),  axis=1)
                
                 #--------------------------------->>>
                ux = tf.expand_dims(tf.cast(tf.linspace(-3, 3, num=self.h), dtype='float32'), axis=0)
                diff_u = abs( Iu - ux)
           
                
                vx = tf.expand_dims(tf.cast(tf.linspace(-3, 3, num=self.h), dtype='float32'), axis=0)
                diff_v = abs( Iv - vx)   
                
            
                #--------------------------------->>>
                if self.method == 'thresholding':
                    diff_u = tf.reshape(diff_u, (-1, self.h)) <= self.eps / 2
                    diff_v = tf.reshape(diff_v, (-1, self.h)) <= self.eps / 2
                #--------------------------------->>>
                elif self.method == 'RBF':
                    diff_u = tf.pow(tf.reshape(diff_u, (-1, self.h)), 2) / self.sigma ** 2
                    diff_v = tf.pow(tf.reshape(diff_v, (-1, self.h)), 2) / self.sigma ** 2
                
                    diff_u = tf.exp(-diff_u)  # Radial basis function
                    diff_v = tf.exp(-diff_v)
                #--------------------------------->>>
                elif self.method == 'inverse-quadratic':
                    diff_u = tf.pow(tf.reshape(diff_u, (-1, self.h)), 2) / self.sigma ** 2
                    diff_v = tf.pow(tf.reshape(diff_v, (-1, self.h)), 2) / self.sigma ** 2
                    
                    
                
                    diff_u = 1 / (1 + diff_u)  # Inverse quadratic
                    diff_v = 1 / (1 + diff_v)
                    
                #--------------------------------->>>
                else:
                    raise Exception(
                        f'Wrong kernel method. It should be either thresholding, RBF,' 
                        f' inverse-quadratic. But the given value is {self.method}.')
            
            
                #--------------------------------->>>    
                diff_u = tf.cast(diff_u, dtype = tf.float32)
                diff_v = tf.cast(diff_v, dtype = tf.float32)
                
                #--------------------------------->>> 
                a = Iy * diff_u
                a = tf.transpose(a)
                
                #--------------------------------->>>
                hists[lth, :, :, ich] = tf.tensordot(a, diff_v, axes=1) 
                
         
        # normalization
        #--------------------------------------------------------------------------------->>>
        divide_x = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum(hists, axis=3), axis=1), axis=1)
        divide_x = tf.expand_dims(divide_x, axis=1)
        divide_x = tf.expand_dims(divide_x, axis=1)
        divide_x = tf.expand_dims(divide_x, axis=1)
        #--------------------------------------------------------------------------------->>>
        hists_normalized = hists / (divide_x + self.EPS)
        return hists_normalized
    
    
        
        
    def compute_histogram_loss(self, input_image, target_image):
        self.target_hist = self.call(target_image)
        self.input_hist = self.call(input_image)
        
        self.histogram_loss = (1/np.sqrt(2.0) * (tf.sqrt(tf.math.reduce_sum( tf.pow(tf.sqrt(self.target_hist) - tf.sqrt(self.input_hist), 2)))) 
                          / self.input_hist.shape[0])
        
        
        return self.histogram_loss
    def get_config(self):
        base_config = super().get_config()

        config = {
                 "h"               : self.h,
                 "insz"              : self.insz,
                 "method"      : self.method ,
                 "sigma"         : self.sigma ,
                 "intensity_scale"                 : self.intensity_scale ,

        }
 
        return {**base_config, **config}
    
    # @classmethod
    # def from_config(cls, config):
    #     sublayer_config = config.pop(("nin_q","nin_k","nin_v","nin_h"))
    #     sublayer = keras.saving.deserialize_keras_object(sublayer_config)
    #     return cls(sublayer, **config)
# ++++++++ +++++++++++++++ +++++++++++++++++++++++++++++++++++++++++++++++ ++++++++++++++++++++++++