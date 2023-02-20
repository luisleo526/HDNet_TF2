import tensorflow as tf
# import tensorflow_graphics as tfg
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# *****************************************************************************************************

def calc_loss(output, y, z_r):
    
    # y refine
    y_masked = tf.compat.v1.where(z_r, y, 0*tf.ones_like(y))
    y_masked_flat_refined = tf.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # output refine
    o_masked = tf.compat.v1.where(z_r, output, 0*tf.ones_like(y))
    o_masked_flat_refined = tf.reshape(o_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # mask refine
    mask_one_refined = tf.compat.v1.where(z_r, tf.ones_like(y), 0*tf.ones_like(y))
    mask_one_flat_refined = tf.reshape(mask_one_refined,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # num of pixels
    numOfPix = tf.reduce_sum(input_tensor=mask_one_flat_refined,axis=1)
    
    d = tf.subtract(o_masked_flat_refined, y_masked_flat_refined)
    d_sum = tf.reduce_sum(input_tensor=tf.square(d),axis=1)
    
    cost = tf.reduce_mean(input_tensor=tf.truediv(d_sum, numOfPix))
    return cost

# *****************************************************************************************************

def calc_loss_normal(output, y_normal,z_refined):

    # gives mean angle error for given output tensor and its ref y
    output_mask = tf.abs(output) < 1e-5
    output_no0 = tf.compat.v1.where(output_mask, 1e-5*tf.ones_like(output), output)
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(output_no0),axis=3)),-1)
    output_unit = tf.divide(output_no0,output_mag)

    z_mask = z_refined[...,0]
    a11 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.square(output_unit),axis=3),mask=z_mask)
    a22 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.square(y_normal),axis=3),mask=z_mask)
    a12 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.multiply(output_unit,y_normal),axis=3),mask=z_mask)

    cos_angle = a12/tf.sqrt(tf.multiply(a11,a22))
    cos_angle_clipped = tf.clip_by_value(tf.compat.v1.where(tf.math.is_nan(cos_angle),-1*tf.ones_like(cos_angle),cos_angle),-1,1)
    # MAE, using tf.acos() is numerically unstable, here use Taylor expansion of "acos" instead
    loss = tf.reduce_mean(input_tensor=3.1415926/2-cos_angle_clipped-tf.pow(cos_angle_clipped,3)/6-tf.pow(cos_angle_clipped,5)*3/40-tf.pow(cos_angle_clipped,7)*5/112-tf.pow(cos_angle_clipped,9)*35/1152)
    return loss

def calc_loss_normal2(output, y_normal,z_refined):
    
    # gives mean angle error for given output tensor and its ref y
    output_mask = tf.abs(output) < 1e-5
    output_no0 = tf.compat.v1.where(output_mask, 1e-5*tf.ones_like(output), output)
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(input_tensor=tf.square(output_no0),axis=3)),-1)
    output_unit = tf.divide(output_no0,output_mag)
    
    z_mask = z_refined[...,0]
    a11 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.square(output_unit),axis=3),mask=z_mask)
    a22 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.square(y_normal),axis=3),mask=z_mask)
    a12 = tf.boolean_mask(tensor=tf.reduce_sum(input_tensor=tf.multiply(output_unit,y_normal),axis=3),mask=z_mask)

    cos_angle = a12/(a11+0.00001)
    loss = tf.reduce_mean(input_tensor=tf.acos(cos_angle))
    return loss


# *****************************************************************************************************

def calc_loss_d_refined_mask(output, y, z_refined):
    
    multiply = tf.constant([IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # mask nonrefine
    mask_one = tf.compat.v1.where(z_refined, tf.ones_like(y), 0*tf.ones_like(y))
    mask_one_flat = tf.reshape(mask_one,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # y refine
    y_masked = tf.compat.v1.where(z_refined, y, 0*tf.ones_like(y))
    y_masked_flat_refined = tf.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    max_y = tf.reduce_max(input_tensor=y_masked_flat_refined,axis=1)
    matrix_max_y = tf.transpose(a=tf.reshape(tf.tile(max_y, multiply), [ multiply[0], tf.shape(input=max_y)[0]]))
    
    # normalize depth
    output_flat = tf.reshape(output,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    output_flat_masked = tf.multiply(output_flat, mask_one_flat)
    
    output_max = tf.reduce_max(input_tensor=output_flat_masked,axis=1)
    matrix_max = tf.transpose(a=tf.reshape(tf.tile(output_max, multiply), [ multiply[0], tf.shape(input=output_max)[0]]))

    output_min = tf.reduce_min(input_tensor=output_flat_masked,axis=1)
    matrix_min = tf.transpose(a=tf.reshape(tf.tile(output_min, multiply), [ multiply[0], tf.shape(input=output_min)[0]]))

    output_unit_flat = tf.truediv(tf.subtract(output_flat_masked,matrix_min),tf.subtract(matrix_max,matrix_min))
    output_unit_flat = tf.multiply(output_unit_flat,matrix_max_y)
    
    # mask refine
    mask_one_refined = tf.compat.v1.where(z_refined, tf.ones_like(y), 0*tf.ones_like(y))
    mask_one_flat_refined = tf.reshape(mask_one_refined,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    # output refine
    output_unit_masked_flat_refined = tf.multiply(output_unit_flat, mask_one_flat_refined)

    # y refine
    y_masked = tf.compat.v1.where(z_refined, y, 0*tf.ones_like(y))
    y_masked_flat_refined = tf.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH])
    
    
    numOfPix = tf.reduce_sum(input_tensor=mask_one_flat_refined,axis=1)
    
    d = tf.subtract(output_unit_masked_flat_refined, y_masked_flat_refined)
    a1 = tf.reduce_sum(input_tensor=tf.square(d),axis=1)
    a2 = tf.square(tf.reduce_sum(input_tensor=d,axis=1))

    cost = tf.reduce_mean(input_tensor=tf.truediv(a1, numOfPix) - (0.5 * tf.truediv(a2, tf.square(numOfPix))))
    return cost
