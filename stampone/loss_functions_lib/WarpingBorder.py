"""
Blends the encoder residual back into the (optionally perspective-warped)
cover image, simulating how the encoded region reattaches to its border
after printing/scanning warp correction.
"""

import tensorflow as tf
import tensorflow_addons as tfa
tf.get_logger().setLevel("DEBUG")

def SSBorder(image_input, input_warped, residual,residual_warped, M,borders):
    if borders == 'no_edge':
        encoded_image = image_input + residual
    elif borders == 'image':
        mask = tfa.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
        encoded_image = residual_warped + input_warped
        encoded_image = tfa.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
        encoded_image += (1-mask) *  tf.roll(image_input, shift=1, axis=0)


    #if borders == 'no_edge':
    #    D_output_real, _ = discriminator(image_input)
    #    D_output_fake, D_heatmap = discriminator(encoded_image)
    #else:
    #    D_output_real, _ = discriminator(input_warped)
    #    D_output_fake, D_heatmap = discriminator(encoded_warped)

    return encoded_image