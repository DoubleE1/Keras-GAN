# calculate inception score for cifar-10 in Keras
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from numpy import ones, expand_dims, log, mean, std, exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.datasets import cifar10, mnist
from skimage.transform import resize
from numpy import asarray
from PIL import Image
import os.path
from os import path
from IPython.display import clear_output

# scale an array of images to a new size
def scale_images(images, new_shape):
  images_list = list()
  for image in images:
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
  return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
      # retrieve images
      ix_start, ix_end = i * n_part, (i+1) * n_part
      subset = images[ix_start:ix_end]
      # convert from uint8 to float32
      print(i, ix_end, ix_start, n_part)
      subset = subset.astype('float32')
      # scale images to the required size
      subset = scale_images(subset, (299,299,3))
      # pre-process images, scale to [-1,1]
      subset = preprocess_input(subset)
      # predict p(y|x)
      p_yx = model.predict(subset)
      # calculate p(y)
      p_y = expand_dims(p_yx.mean(axis=0), 0)
      # calculate KL divergence using log probabilities
      kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
      # sum over classes
      sum_kl_d = kl_d.sum(axis=1)
      # average over images
      avg_kl_d = mean(sum_kl_d)
      # undo the log
      is_score = exp(avg_kl_d)
      # store
      scores.append(is_score)
      # print(i)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

# load cifar10 images
(images, classes), (_, _) = cifar10.load_data()

# Extract dogs and cats
X_cats = images[(classes == 3).flatten()]
X_dogs = images[(classes == 5).flatten()]
images = np.vstack((X_cats, X_dogs))

images = images[:1000]
# shuffle images
shuffle(images)
print('loaded', images.shape)

is_avg, is_std = calculate_inception_score(images)
print('average inception score:', is_avg, 'standard deviation inception scores:', is_std)
