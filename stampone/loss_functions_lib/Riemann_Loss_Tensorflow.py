


"""
@--23.10.2023--@
Author: github/farhadsh1992
INFO:
	- Riemann Loss TF version: compares covariance matrices of the cover
	  and generated images on the Riemannian manifold of SPD matrices,
	  used as an additional GAN training loss.
    - REF:



LAST_UPDATE:
"""



import tensorflow as tf
import keras
import warnings
tf.get_logger().setLevel("INFO")
warnings.filterwarnings('ignore')
#################################################################################
#######                                #######
#################################################################################

@keras.utils.register_keras_serializable(package="Riemann_Loss")
class Riemann_Loss(tf.keras.losses.Loss):
    """
    INPUTS:
    OUTPUTS:
    INFO:
        Tensorflow version of the Riemann loss function for GAN models.  
    
    """
    def __init__(self, batch:int=1, image_size:int=256):
        super(Riemann_Loss, self).__init__()
        
        self.batch = batch
        self.image_size = image_size

        self.reshape_fn = keras.layers.Reshape(target_shape=(self.image_size*self.image_size, 3))
        # self.distance_fun = tf.keras.layers.Lambda(lambda AB: tf.map_fn(self.distance_riemann, AB))
        self.distance_fun = keras.layers.Lambda(lambda AB: self.distance_riemann(AB))
        # self.distance_fun = tf.map_fn(lambda AB: self.distance_riemann(AB))

    def tf_cov(self, x:tf.Tensor)->tf.Tensor:    
        '''
        mean_x = tf.reduce_mean(x, axis=0)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        '''
    
        mean_x = keras.ops.mean(x, axis=0)
        med_x = x-mean_x
   
        cov_xx = tf.map_fn(lambda med_x: tf.matmul(tf.transpose(med_x), med_x)/tf.cast(tf.shape(med_x)[0], tf.float32), med_x)

        return cov_xx
    def distance_riemann(self, AB)->tf.Tensor:
        A, B = AB
        A = self.tf_cov(A*255)
        B = self.tf_cov(B*255) 
        B, A = tf.abs(B +1e-12), tf.abs(A +1e-12)
        
        c = A*tf.math.reciprocal(B) 
        dist = tf.math.real(tf.linalg.trace((tf.math.log(c)**2))) 
        return dist
    
    # @tf.function(jit_compile=True)
    def __call__(self, original_image:tf.Tensor, generated_image:tf.Tensor, fake_output_disc:tf.Tensor=None)->tf.Tensor:
        
        if original_image.shape[3] == 1:
            ## Gray image to RGB image (real and fake)
            img = tf.image.grayscale_to_rgb(original_image)
            generated = tf.image.grayscale_to_rgb(generated_image)
        else:
            img = original_image
            generated = generated_image
            
        ## reshape real and fake from (B, w, h, 3) into (B, w*h, 3)
        img =  self.reshape_fn(img)
        generated =  self.reshape_fn(generated)
        
        ## compute distance in riemann space
        loss = self.distance_fun((img, generated))
       

        ## Normalize the loss value
        loss_op = loss/tf.cast((self.batch*1), dtype="float32")

        ## (????)
        # loss_c = torch.nn.functional.binary_cross_entropy_with_logits(tf.ones_like(fake_output_disc), fake_output_disc)
        out = keras.ops.mean(loss_op) #/5000 + loss_c ## (????)
        
        return out 









#################################################################################
#######                                #######
#################################################################################

def tf_cov(x:tf.Tensor)->tf.Tensor:    
    '''
    mean_x = tf.reduce_mean(x, axis=0)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    '''
    
    mean_x = tf.reduce_mean(x, axis=0)
    med_x = x-mean_x
    # cov_xx = tf.matmul(tf.transpose(med_x), med_x)/tf.cast(tf.shape(med_x)[0], tf.float32)
    z = tf.transpose(med_x, perm=[0, 2, 1])
    cov_xx = tf.matmul(z * med_x)
    return cov_xx
#################################################################################
def distance_riemann(AB)->tf.Tensor:
    A, B = AB
    A = tf_cov(A*255)
    B = tf_cov(B*255) 
    B, A = tf.abs(B +1e-12), tf.abs(A +1e-12)
        
    c = A*tf.math.reciprocal(B) 
    dist = tf.math.real(tf.linalg.trace((tf.math.log(c)**2))) 
    return dist
#################################################################################



