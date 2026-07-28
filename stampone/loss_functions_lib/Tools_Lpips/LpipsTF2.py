






"""
@--22.12.2023--@
Author: github/farhadsh1992
INFO:
    LPIPS wrapper (TF1 frozen-graph and TFLite variants) computing
    learned perceptual similarity between the cover and encoded images.
    -ref:

LAST_UPDATE:
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import jax
import warnings

tf.get_logger().setLevel("INFO")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
#############################################################################################

MODEL_DIR = ("./loss_functions_lib/lpips_model/net-lin_alex_v0.1.pb")

#############################################################################################
#####                                                    #####
#############################################################################################

@keras.utils.register_keras_serializable(package="Lpips_TF1")
class Lpips_TF1(keras.models.Model):
    def __init__(self)-> None:
        super(Lpips_TF1, self).__init__()
        self.MODEL_DIR = MODEL_DIR
        
        self.build()
    def build(self)-> None:
        with tf.io.gfile.GFile(self.MODEL_DIR, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        self.load_frozen = self.wrap_frozen_graph(graph_def, inputs=['0:0', '1:0'], outputs='Reshape_10:0')
    
    def wrap_frozen_graph(self, graph_def, inputs, outputs)-> None:
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
              tf.nest.map_structure(import_graph.as_graph_element, inputs),
              tf.nest.map_structure(import_graph.as_graph_element, outputs))
    
    # @tf.function(jit_compile=True)
    # @jax.jit
    def __call__(self, input0:tf.Tensor, input1:tf.Tensor)->tf.Tensor:
        
        batch_shape = tf.shape(input0)[:-3]
        # input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
        # input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
        # NHWC to NCHW
        input0 = tf.transpose(input0, [0, 3, 1, 2])
        input1 = tf.transpose(input1, [0, 3, 1, 2])
        # normalize to [-1, 1]
        input0 = input0 * 2.0 - 1.0
        input1 = input1 * 2.0 - 1.0
        
        distance1 = self.load_frozen(input0,input1)
        
        if distance1.shape.ndims == 4:
            distance = tf.squeeze(distance1, axis=[-3, -2, -1])
        # reshape the leading dimensions
        distance2 = tf.reshape(distance, batch_shape)
        distance = keras.ops.mean(distance2)

        del(input0)
        del(input1)
        del(distance1)
        del(distance2)

        return distance
    

@keras.utils.register_keras_serializable(package="Lpips_keras3")
class Lpips_keras3(keras.layers.Layer):
    def __init__(self, filepath:str='./loss_functions_lib/lpips_model/lpips_model', 
                 use_normalize:bool=False, use_resize:bool=False)->tf.Tensor:
        super(Lpips_keras3, self).__init__()

        # self.lpips_model = keras.models.load_model('./loss_functions_lib/lpips_model/lpips_model.h5')
        self.lpips_model = keras.layers.TFSMLayer('./loss_functions_lib/lpips_model/lpips_model2', call_endpoint='Reshape_10:0')

        self.use_normalize = use_normalize
        self.use_resize = use_resize
        
    @tf.function(jit_compile=True)
    def __call__(self, input0:tf.Tensor, input1:tf.Tensor) -> tf.Tensor:


        if self.use_normalize:
            input0 = normalize_fixed(input0, current_range=[-1,1], normed_range=[0,1])
            input1 = normalize_fixed(input1, current_range=[-1,1], normed_range=[0,1])
        if self.use_resize:
            pass

        # NHWC to NCHW
        xinput0 = tf.transpose(input0, [0, 3, 1, 2])
        xinput1 = tf.transpose(input1, [0, 3, 1, 2])

        error = self.lpips_model(xinput0, xinput1)
        error = keras.ops.mean(error)

        del(xinput0)
        del(xinput1)
        return error
###################################################################################################
def normalize_fixed(x, current_range, normed_range):
    # current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    current_min, current_max = current_range[0], current_range[1]

    # normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    normed_min, normed_max = normed_range[0], normed_range[1]

    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed