







"""
Total variation loss: penalizes high-frequency pixel-to-pixel differences
in the encoded image to encourage smoother, less noisy output.
"""

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
tf.get_logger().setLevel("DEBUG")






# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Total_Variation_Loss():
    def __init__(self):
        super(Total_Variation_Loss, self).__init__()
        
        
    def high_pass_x_y(self, image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    def compute(self, image):
        x_deltas, y_deltas = self.high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))    
# ----------------------------------------------------------------------------------------------------------------------