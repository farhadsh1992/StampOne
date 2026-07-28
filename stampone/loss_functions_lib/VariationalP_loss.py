"""
@--19.08.2022--@
Author: github/farhadsh1992
INFO:
  Variational/KL-divergence loss experiment (based on the TF CVAE
  tutorial) for regularizing the message embedding distribution.
  - https://www.tensorflow.org/tutorials/generative/cvae
LAST_UPDATE:
"""

import tensorflow as tf
import numpy as np
import sklearn.datasets as datasets

from stampone.FarhadCV.Tools import tcolors
import cv2
tf.get_logger().setLevel("DEBUG")

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

class transfer_vector_xy():
  def __init__(self,batch, height, width, channel, values):
    super(transfer_vector_xy, self).__init__()
      
    self.batch = batch
    self.height = height
    self.width = width
    self.channel = channel
    self.values = values
      
  def call(self, a_vector_message2D):
    xy_batch = []
    
    for i in range(self.height):
      for j in range(self.width):
        for o in range(self.channel):
          if a_vector_message2D[i,j,o] >= self.values:
            xy_batch.append([i, j])
    
    
    xy_batch = (np.array(xy_batch)/self.height)
    return xy_batch
    

class KL1(tf.keras.layers.Layer):
  def __init__(self, batch, height, width, channel):
    super(KL1, self).__init__()
    self.batch = batch
    # data, _ = datasets.make_moons(moon_n, noise=0.05)
    transfer_router = transfer_vector_xy(batch, height, width, channel=1, values=0.45)

    self.transfer =tf.keras.layers.Lambda(lambda x: tf.map_fn(transfer_router.call, x))
    # tf.keras.layers.Lambda(lambda number: datasets.make_moons(number, noise=0.05)

    image = cv2.imread("/media/ssd2_data/Farhad_Sirin_ssd2/0023_02_04__AttentionVNet_02/concentric-circles-rings.jpg")
    image = cv2.resize(image, (16, 16))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = np.expand_dims(image, axis=2)

    image = np.where((image < 128), 0, image)
    image = np.where((image > 128), 1, image)

    image_batch = []
    for i in range(batch):
      image_batch.append((image)-0.5) 
    self.image_batch = np.array(image_batch)
    self.image_batch = tf.cast(self.image_batch, dtype="float32")

  def reparameterize(self, z_emb):
    # mean, logvar = tf.split(x_logit, num_or_size_splits=2, axis=1)
    mean = z_emb
    logvar = z_emb
    eps = tf.random.normal(shape=mean.shape)
    z =  eps * tf.exp(logvar * .5) + mean
    return z, mean, logvar

  def compute_loss(self, embeddding):
    
    # x_logit = embedding[:,:,:,:3]
    # #x_logit = tf.image.rgb_to_grayscale(x_logit)
    # x_logit = tf.cast(x_logit, dtype="float32")

    # xy_batch = self.transfer(x_logit)

    # xy_batch = tf.cast(xy_batch, dtype="float32")

    # print(tcolors.RED, xy_batch.shape, tcolors.ENDC)

    # number = xy_batch.shape[1]
    
    # data_batch = []
    # for i in range(self.batch):
    #   data, _ = datasets.make_moons(number, noise=0.05)
    #   data_batch.append(data)
    # data_batch = tf.cast(np.array(data_batch), dtype="float32")
    # z, mean, logvar = self.reparameterize(z_emb)
    mean, logvar = embeddding, embeddding

    
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_Sprime, labels=self.image_batch)
    # print(tcolors.RED, "cross_ent" ,cross_ent.shape, tcolors.ENDC)

    # logpx_z = -tf.reduce_sum(cross_ent, axis=3)
    # logpz = log_normal_pdf(z, 0., 0.)
    # logqz_x = log_normal_pdf(z, mean, logvar)
    # print(tcolors.RED, "logpx_z" ,logpx_z.shape, tcolors.ENDC)
    # print(tcolors.RED, "logpz" ,logpz.shape, tcolors.ENDC)
    # print(tcolors.RED, "logqz_x" ,logqz_x.shape, tcolors.ENDC)

    out = -0.5 * tf.reduce_sum(1 + logvar - tf.math.pow(mean, 2) - tf.math.exp(logvar))

    # out = -tf.reduce_mean(logpx_z + logpz - logqz_x)

    return out
