

"""
@--22.12.2023--@
Author: github/farhadsh1992
INFO:
    Keras 3-compatible LPIPS perceptual-similarity metric: runs a frozen
    VGG feature extractor plus learned linear weights to score perceptual
    distance between the cover and encoded images.
    -ref:

LAST_UPDATE:
"""

import tensorflow as tf
import keras
import warnings
# tf.get_logger().setLevel("INFO")
warnings.filterwarnings('ignore')


@keras.utils.register_keras_serializable(package="learned_perceptual_metric_model")
class learned_perceptual_metric_model(keras.models.Model):
    def __init__(self, path:str=None)->None:
        super(learned_perceptual_metric_model,self).__init__()

        self.resize_layer_1 = tf.keras.layers.Resizing(height=64,width=64)
        self.resize_layer_2 = tf.keras.layers.Resizing(height=64,width=64)


        self.images_size:int = 256
        # self.LPIPS_Model:keras.models.Model = tf.keras.models.load_model('./loss_functions_lib/lpipstf/keras_lpips_64size.h5')
        self.build(input_shape=(256,256))


    def build(self, input_shape):
        image_size = input_shape[0]
        ## initialize all models
        net = keras.models.load_model('./loss_functions_lib/lpipstf/vgg_ckpt_fn.h5')
        lin = keras.models.load_model('./loss_functions_lib/lpipstf/linear_ckpt_fn2.h5')

        ## merge two model
        input1 = keras.layers.Input(shape=(image_size, image_size, 3), dtype='float32', name='input1')
        input2 = keras.layers.Input(shape=(image_size, image_size, 3), dtype='float32', name='input2')

        ## run vgg model first
        net_out1 = net(input1)
        net_out2 = net(input2)

        ## normalize
        net_out1 = [tf.keras.layers.Normalization()(t) for t in net_out1]
        net_out2 = [tf.keras.layers.Normalization()(t) for t in net_out2]
        
        # subtract
        diffs = [keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]), 
                                     name=f"subtrac_{i}")([net_out1[i], net_out2[i]]) for i, t2 in enumerate(net_out1)]

        # run on learned linear model
        lin_out = lin(diffs)

        # take spatial average: list([N, 1], [N, 1], [N, 1], [N, 1], [N, 1])
        lin_out = [keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=False), 
                                       name=f"special_{i}")(t) for i, t in enumerate(lin_out)]

        # take sum of all layers: [N, 1]
        lin_out = keras.layers.Lambda(lambda x: tf.add_n(x), output_shape=(None), name="lambda_last")(lin_out)
    
        # squeeze: [N, ]
        # lin_out = Lambda(lambda x: tf.squeeze(x, axis=-1))(lin_out)
        self.LPIPS_Model = keras.models.Model(inputs=[input1, input2], outputs=lin_out, name='lpips_metrics')
    # @tf.function(jit_compile=True)
    def __call__(self, input_1:tf.Tensor , input_2:tf.Tensor)->tf.Tensor:

        # input_1 = self.resize_layer_1(input_1)
        # input_2 = self.resize_layer_1(input_2)

        error = self.LPIPS_Model([input_1, input_2])
        error = keras.ops.abs(error)
        error = keras.ops.mean(error)
        return error