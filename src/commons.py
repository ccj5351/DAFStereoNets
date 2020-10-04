from src.utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
from tensorflow.python.framework import ops

""" added by CCJ for segmentation-aware convolution """
# images : in shape [N, H, W, C];
# k : filter_size, corresponding to window in size [k, k];
# r : equivalent to rate in dilated (a.k.a. Atrous) convolutions.
def im2col(images, k = 3, r = 1, scope = 'im2col'):
    # tf.extract_image_patches: Extract patches from images and put them in the "depth" output dimension.
    image_patches = tf.extract_image_patches(
            images = images, 
            ksizes = [1,k,k,1],
            strides = [1,1,1,1], 
            rates = [1,r,r,1], # equivalent to rate in dilated (a.k.a. Atrous) convolutions.
            padding = 'SAME',
            name=scope)
    return image_patches # in shape [N, H, W, C*(k*K)];

def dist_loss(parity, distance, labels, ignore_label = 255, k = 3, alpha = 0.5, beta = 2.0, scope = 'loss'):

    label2col_patch = im2col(labels, k, 1, 'label2col_patch') # in shape [N, H, W, k*k]
    ignore = tf.cast(label2col_patch < ignore_label , tf.float32) # in shape [N,H,W,k*k]
    """for debugging """
    #ignore = tf.Print(input_ = ignore, data = [tf.reduce_mean(ignore),], message = "[*****] mean of ignore : ", first_n = 3)
    dist_same_label = ignore * tf.cast(tf.maximum(distance-alpha, 0.0), tf.float32) # in shape [N,H,W,k*k]
    dist_diff_label = ignore * tf.cast(tf.maximum(beta-distance, 0.0), tf.float32) # in shape [N,H,W,k*k]
    loss = tf.where(condition = parity, x=dist_same_label, y=dist_diff_label) # in shape [N,H,W,k*k]
    loss = tf.cast(tf.reduce_sum(loss, axis=-1), tf.float32) # in shape [N,H,W]
    print ("loss = {}".format(loss))
    loss_avg = tf.reduce_mean(loss)
    
    return loss_avg


def im2parity(labels, k = 3, r = 1, scope = 'im2parity'):
    #N,H,W,_ = tf.shape(labels) # Error: Tensor objects are not iterable when eager execution is not;
    N = tf.shape(labels)[0]
    H = tf.shape(labels)[1]
    W = tf.shape(labels)[2]
    im2col_patch = im2col(labels, k, r, 'im2col_patch') # in shape [N, H, W, k*k]
    """ tile """
    #im2col_tile =  tf.tile(labels, [1,1,1,k*k])
    """ tf.equal supports broadcasting """
    im2col_tile =  tf.reshape(labels, [N,H,W,1])
    #parity = tf.cast( tf.equal(im2col_patch, im2col_tile), tf.float32)
    parity = tf.equal(im2col_patch, im2col_tile)
    #print ("parity.shape = {}".format(parity.shape))
    return parity # in shape [N,H,W,k*k]


def im2dist_L1(images, k = 3, r = 1, scope = 'im2col'):
    #N,H,W,C = tf.shape(images)
    N = tf.shape(images)[0]
    H = tf.shape(images)[1]
    W = tf.shape(images)[2]
    C = tf.shape(images)[3]
    """ C-chunk is continuous in memory """
    # in shape [N, H, W, C*(k*k)], i.e, (x^1_1,...,x^1_c), (...), (x^{k*k}_1, ..., x^{k*k}_c);
    im2col_patch = im2col(images, k, r, 'im2col_patch') 
    #print ('im2col_patch shape = {}'.format(im2col_patch.shape))
    im2col_patch = tf.reshape(im2col_patch, [N,H,W,k*k,C])
    """ tile """
    im2col_tile = tf.reshape(tf.tile(images, [1,1,1,k*k]), [N,H,W,k*k,C])
    l1_dist = tf.reduce_sum(tf.abs(im2col_patch - im2col_tile), axis = -1)
    #print ("l1_dist.shape = {}".format(l1_dist.shape))
    return l1_dist # in shape [N,H,W,k*k]

def im2dist_L2(images, k = 3, r = 1, scope = 'im2col'):
    N,H,W,C = tf.shape(images)
    # in shape [N, H, W, C*(k*k)], i.e, (x^1_1,...,x^1_c), (...), (x^{k*k}, ..., x^{k*k}_c);
    im2col_patch = im2col(images, k, r,'im2col_patch')
    im2col_patch = tf.reshape(im2col_patch, [N,H,W,k*k, C])
    im2col_tile = tf.reshape(tf.tile(images, [1,1,1,k*k]), [N,H,W,k*k,C])
    l2_dist = tf.sqrt(tf.sum(tf.square(im2col_patch - im2col_tile), axis = -1))
    return l2_dist # in shape [N,H,W,k*k]

def conv2d(x, f, k, s=1, d=1, scope='conv', act=tf.nn.relu, bn=True, train=True):
      with tf.variable_scope(scope):
        """ tf.get_variable() : Gets an existing variable with these parameters 
            or create a new one.
        """
        w = tf.get_variable( name = "weights", shape = [k,k,x.shape[-1],f],
                initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))

        b = tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        o = tf.nn.conv2d(
                input = x, # a 4-D tensor.
                filter = w, # a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels];
                strides = [1, s, s, 1], 
                padding = 'SAME', 
                #dilation factor for each dimension of input. If set to k > 1, there will be k-1 skipped cells 
                #between each filter element on that dimension. Dilations in the batch and depth dimensions must be 1.
                dilations=[1,d,d,1])

        """ This is (mostly) a special case of tf.add where bias is 
            restricted to 1-D. Broadcasting is supported, so value 
            may have any number of dimensions. Unlike tf.add, 
            the type of bias is allowed to differ from value 
            in the case where both types are quantized.
        """
        o = tf.nn.bias_add(value = o, bias = b)
        if bn == True:
          """ training: Either a Python boolean, or a TensorFlow boolean 
              scalar tensor (e.g. a placeholder). Whether to return the 
              output in training mode (normalized with statistics of 
              the current batch) or in inference mode (normalized with 
              moving statistics). 
              NOTE: make sure to set this parameter correctly, or else 
              your training/inference will not work properly.
          """
          o = tf.layers.batch_normalization(inputs = o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
      return o
      
def dconv2d(x, f, k, s=1, d=1, scope='conv', act=tf.nn.relu, bn=True, train=True):
      with tf.variable_scope(scope):
        w = tf.get_variable("weights",[k,k,x.shape[-1],f],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b = tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        """ Atrous convolution (a.k.a. convolution with holes or dilated convolution);"""
        o = tf.nn.atrous_conv2d(value = x, filters = w, rate = d, padding = 'SAME')
        o = tf.nn.bias_add(o, b)
        if bn == True:
          o = tf.layers.batch_normalization(o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
      return o      
    
def conv3d(x, f, k, s=1, d=1, scope='conv', act=tf.nn.relu, bn=True, train=True 
        #, reuse=None
        ):
      with tf.variable_scope(scope):
        w = tf.get_variable("weights",[k,k,k,x.shape[-1],f],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b = tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        o = tf.nn.conv3d(x, w, [1, s, s, s, 1], 'SAME', dilations=[1,d,d,d,1])
        o = tf.nn.bias_add(o, b)
        if bn == True:
          o = tf.layers.batch_normalization(o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
      return o

def deconv3d(x, f, k, s=1, scope='conv', act=tf.nn.relu, bn=True, train=True):
      with tf.variable_scope(scope):
        w=tf.get_variable("weights",[k,k,k,f,x.shape[-1]],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b=tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        outputShape=[tf.shape(x)[0],tf.shape(x)[1]*s,tf.shape(x)[2]*s,tf.shape(x)[3]*s,f]
        o = tf.nn.conv3d_transpose(x, w, outputShape, [1, s, s, s, 1], 'SAME')
        o = tf.nn.bias_add(o, b)
        if bn == True:
          o = tf.layers.batch_normalization(o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
      return o

def deconv2d(x, f, k, s=1, scope='conv', act=tf.nn.relu, bn=True, train=True):
      with tf.variable_scope(scope):
        w=tf.get_variable("weights",[k,k,f, x.shape[-1]],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b=tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        output_shape = [tf.shape(x)[0],tf.shape(x)[1]*s,tf.shape(x)[2]*s,f]
        o = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, s, s, 1], padding='SAME')
        o = tf.nn.bias_add(o, b)
        if bn == True:
          o = tf.layers.batch_normalization(o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
        return o

#def conv2d_block(x, f, k, s=1, d=1, scope1='a', scope2='b', act=None, bn=False, train=True):
""" updated by CCJ: change act = None, bn = False to act = tf.nn.relu, bn = True """
def conv2d_block(x, f, k, s=1, d=1, scope1='a', scope2='b', act= tf.nn.relu, bn=True, train=True):
      conv1 = conv2d(x, f, k, s, d, scope=scope1, act=act, bn=bn, train=train)
      print('|----> ' + conv1.name + ' --> ' + str(conv1.shape))
      conv2 = conv2d(conv1, f, k, 1, d, scope=scope2, act=act, bn=bn, train=train)
      print('|----> ' + conv2.name + ' --> ' + str(conv2.shape))
      return conv2

#def conv3d_block(x, f, k, s=1, d=1, scope1='a', scope2='b', train=True):
""" updated by CCJ: added act = tf.nn.relu, bn = True """
def conv3d_block(x, f, k, s=1, d=1, scope1='a', scope2='b', act=tf.nn.relu, bn=True,  train=True):
      conv1 = conv3d(x, f, k, s, d, scope=scope1, act = act, bn = bn, train=train)
      print('|----> ' + conv1.name + ' --> ' + str(conv1.shape))
      conv2 = conv3d(conv1, f, k, 1, d, scope=scope2,  act= act, bn = bn, train=train)
      print('|----> ' + conv2.name + ' --> ' + str(conv2.shape))
      return conv2

def loss_l1(x,y,valid):
      #l1_loss = tf.reduce_sum(tf.multiply(tf.abs(y-x),valid))# / tf.reduce_sum(valid)
      l1_loss = tf.reduce_sum(tf.multiply(tf.abs(y-x),valid))/tf.reduce_sum(valid)
      return l1_loss

def upsampling_block(bottom, skip_connection, input_channels, output_channels, skip_input_channels):
    with tf.variable_scope("deconv") as scope:
        deconv = deconv2d(bottom, output_channels, 4, 2, scope)
    with tf.variable_scope("predict") as scope:
        predict = conv2d(bottom, 1, 3, 1, scope=scope, bn=False, act=None)
        #tf.summary.histogram("predict", predict)
    with tf.variable_scope("up_predict") as scope:
        upsampled_predict = deconv2d(predict, 1, 4, 2, scope=scope, bn=False, act=None)
    with tf.variable_scope("concat") as scope:
        concat = conv2d(tf.concat([skip_connection, deconv, upsampled_predict], axis=3), output_channels, 3, 1, scope=scope, bn=False, act=None)
    return concat, predict
    
def correlation_map(x, y, max_disp):
    corr_tensors = []
    y_shape = tf.shape(y)
    for i in range(-max_disp, 0, 1):
        shifted = tf.pad(tf.slice(y, [0] * 4, [-1, -1, y_shape[2] + i, -1]),
                         [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
        corr_tensors.append(corr)
    for i in range(max_disp + 1):
        shifted = tf.pad(tf.slice(x, [0, 0, i, 0], [-1] * 4),
                         [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
        corr_tensors.append(corr)
    return tf.transpose(tf.stack(corr_tensors),perm=[1, 2, 3, 0])
    
def conv3d_r(x, f, k, s=1, scope='conv', act=tf.nn.relu, bn=True, train=True,
        #, reuse=None
        ):
      with tf.variable_scope(scope):
        w = tf.get_variable("weights",[k[0],k[1],k[2],x.shape[-1],f],initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b = tf.get_variable("biases", f, initializer=tf.constant_initializer(0.0))
        o = tf.nn.conv3d(x, w, [1, s, s, s, 1], 'SAME')
        o = tf.nn.bias_add(o, b)
        if bn == True:
          o = tf.layers.batch_normalization(o, training=train, name="bn")
#        else:
#          print ('WARNING: BN disabled')
        if act is not None:
          o = act(o)
#        else:
#          print ('WARNING: No activation')
      return o

      
def resize_bilinear(src, scale):

      input_spatial_rank = src.shape.ndims-2
      assert input_spatial_rank in (2, 3), \
        "linearly interpolation layer can only be applied to " \
        "2D/3D images (4D or 5D tensor)."
      shape = tf.shape(src)

      if input_spatial_rank == 2:
        size = [shape[0], shape[1], shape[2], shape[3]]
        new_size = [shape[1]*scale, shape[2]*scale]
        return tf.image.resize_bilinear(src, new_size)

      size = [shape[0], shape[1], shape[2], shape[3], shape[4]]
      new_size = [shape[1]*scale, shape[2]*scale, shape[3]*scale]

      b_size, x_size, y_size, z_size, c_size = size
      x_size_new, y_size_new, z_size_new = new_size

      if (x_size == x_size_new) and (y_size == y_size_new) and (z_size == z_size_new):
        # already in the target shape
        return src

      # resize y-z
      squeeze_b_x = tf.reshape(src, [-1, y_size, z_size, c_size])
      resize_b_x = tf.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new])
      resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

      # resize x
      # first reorient
      reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
      # squeeze and 2d resize
      squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
      resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
      resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

      output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
      return output_tensor    

def scale_pyramid(img, num_scales, type='bilinear'):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = tf.to_int32(h / ratio)
        nw = tf.to_int32(w / ratio)
        if type == 'nearest':
          scaled_imgs.append(tf.image.resize_nearest_neighbor(img, [nh, nw]))
        if type == 'bilinear':
          scaled_imgs.append(tf.image.resize_bilinear(img, [nh, nw]))

    return scaled_imgs

def build_warping_coords(self, disparity):

        h = tf.to_int32(tf.shape(disparity)[1])
        w = tf.to_int32(tf.shape(disparity)[2])
        b = tf.to_int32(tf.shape(disparity)[0])

        tb = tf.expand_dims(tf.expand_dims(tf.linspace(0.,tf.cast(b, tf.float32),b),-1),-1)
        bcoords = tf.expand_dims(tf.tile(tb, tf.stack([1,h,w])),-1)
        tx = tf.expand_dims(tf.linspace(0.,tf.cast(w, tf.float32),w),0)
        xcoords = tf.expand_dims(tf.tile(tx, tf.stack([h,1])), -1)
        ty = tf.expand_dims(tf.linspace(0.,tf.cast(h, tf.float32),h),1)
        ycoords = tf.expand_dims(tf.tile(ty, tf.stack([1,w])), -1)  
        meshgrid = tf.expand_dims(tf.concat([xcoords,ycoords],-1),0)
        coords = tf.concat([disparity, tf.zeros_like(disparity)], -1)
        output = tf.concat([bcoords,meshgrid + coords], -1)       
        return output

def backward_warping(imgs, coords):
      shape = coords.get_shape().as_list()
      coord_b, coords_x, coords_y = tf.split(coords, [1, 1, 1], axis=3)

      x0 = tf.floor(coords_x)
      x1 = x0 + 1
      y0 = tf.floor(coords_y)
      y1 = y0 + 1

      y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
      x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
      zero = tf.zeros([], dtype='float32')

      x0_safe = tf.clip_by_value(x0, zero, x_max)
      y0_safe = tf.clip_by_value(y0, zero, y_max)
      x1_safe = tf.clip_by_value(x1, zero, x_max)
      y1_safe = tf.clip_by_value(y1, zero, y_max)

      ## bilinear interp weights, with points outside the grid having weight 0
      wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
      wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
      wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
      wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

      im00 = tf.cast( tf.gather_nd(imgs, tf.cast(tf.concat([coord_b, y0_safe, x0_safe], -1), 'int32')), 'float32')
      im01 = tf.cast( tf.gather_nd(imgs, tf.cast(tf.concat([coord_b, y0_safe, x1_safe], -1), 'int32')), 'float32')
      im10 = tf.cast( tf.gather_nd(imgs, tf.cast(tf.concat([coord_b, y1_safe, x0_safe], -1), 'int32')), 'float32')
      im11 = tf.cast( tf.gather_nd(imgs, tf.cast(tf.concat([coord_b, y1_safe, x1_safe], -1), 'int32')), 'float32')

      w00 = wt_x0 * wt_y0
      w01 = wt_x0 * wt_y1
      w10 = wt_x1 * wt_y0
      w11 = wt_x1 * wt_y1

      output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
      ])

      return output      

def generate_image_left(batch_size, img, disp):
    coords = build_warping_coords(batch_size, -disp)
    return backward_warping(img, coords)

def generate_image_right(batch_size, img, disp):
    coords = build_warping_coords(batch_size, disp)
    return backward_warping(img, coords)  
    
def soft_argmin(volume, dmax, batch):
    h = tf.to_int32(tf.shape(volume)[2])
    w = tf.to_int32(tf.shape(volume)[3])

    softmax = tf.nn.softmax(volume*-1., 1)
    disparities = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.linspace(0.,tf.cast(dmax, tf.float32),dmax),0),-1),-1),-1), tf.stack([batch,1,h,w,1]))   
    disparity = tf.reduce_sum(tf.multiply(softmax, disparities),1) 
    #print('|-------> Disparity map --> before squeeze : ' + str(disparity.shape))
    #""" added by CCJ, squeeze the last channel, due to its size of 1 """
    #disparity = tf.squeeze(disparity, axis = -1)
    #print('|-------> Disparity map --> after squeeze : ' + str(disparity.shape))
    print('|-------> Disparity map -->  ' + str(disparity.shape))
    return disparity
