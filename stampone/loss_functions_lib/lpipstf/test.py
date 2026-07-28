"""
Vendored smoke test from alexlee-gk/lpips-tensorflow: loads the exported
LPIPS TF model and prints its distance for the library's example image
pairs.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models.lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image


image_size = 64
model_dir = './models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

# official pytorch model metric value
# ex_ref.png <-> ex_p0.png: 0.569
# ex_ref.png <-> ex_p1.png: 0.422
image_fn1 = './imgs/ex_ref.png'
image_fn2 = './imgs/ex_p0.png'
image_fn3 = './imgs/ex_p1.png'

# images should be RGB normalized to [0.0, 255.0]
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
image3 = load_image(image_fn3)

batch_ref = tf.concat([image1, image1], axis=0)
batch_inp = tf.concat([image2, image3], axis=0)
metric = lpips([batch_ref, batch_inp])
print(f'ref shape: {batch_ref.shape}')
print(f'inp shape: {batch_inp.shape}')
print(f'lpips metric shape: {metric.shape}')
print(f'ref <-> p0: {metric[0]:.3f}')
print(f'ref <-> p1: {metric[1]:.3f}')
