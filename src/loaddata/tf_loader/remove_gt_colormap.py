# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np
import sys


from PIL import Image

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_gt_folder',
                           './VOCdevkit/VOC2012/SegmentationClass',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.app.flags.DEFINE_string('output_dir',
                           './VOCdevkit/VOC2012/SegmentationClassRaw',
                           'folder to save modified ground truth annotations.')
""" added by CCJ """
import cv2


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.

  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.gfile.Open(filename, mode='w') as f:
      #NOTE: maybe this 
      pil_image.save(f, 'PNG')


""" added by CCJ """
def color_map_info(palette):
    labels = [
          'background', #0
          'aeroplane', #1
          'bicycle', #2
          'bird', #3
          'boat', #4
          'bottle', #5
          'bus', #6
          'car', #7
          'cat', #8
          'chair', #9
          'cow', #10
          'diningtable', #11
          'dog', #12
          'horse', #13
          'motorbike', #14
          'person', #15
          'pottedplant', #16
          'sheep', #17
          'sofa', #18
          'train', #19
          'tv/monitor', #20
          "void/unlabelled", #255
          ] 
    print 'class colormap and palette = {r,g,b}'
    for i in range(0,21*3,3):
        print '# {:>3d}: {:<20} (R,G,B) = {},{},{}'.format(i/3, labels[i/3], palette[i], palette[i+1],palette[i+2])
    i = 255*3
    print '# {:>3d}: {:<20} (R,G,B) = {},{},{}'.format(i/3, labels[21], palette[i], palette[i+1],palette[i+2])




""" 
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""

# see https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae;
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3 # right-shif 3 bits
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz_1_column():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 
            'tvmonitor', 'void']

    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map(N=256)
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
        array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]
    imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()


# added by CCJ:
""" arrange these 21 classes to 2D matrix with 3 rows and 7 columns"""
def color_map_viz(fname = None):
    labels = ['B-ground', 'Aero plane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 
            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Dining-Table', 'Dog', 'Horse',
            'Motorbike', 'Person', 'Potted-Plant', 'Sheep', 'Sofa', 'Train', 
            'TV/Monitor', 'Void/Unlabelled']

    nclasses = 21
    row_size = 80
    col_size = 250
    cmap = color_map(N=256)

    """ arrange these 21 classes to 2D matrix with 3 rows and 7 columns"""
    r = 3
    c = 7
    delta = 10
    array = np.empty((row_size*(r+1), col_size*c, cmap.shape[1]), dtype=cmap.dtype)
    fig=plt.figure()
    for r_idx in range(0,r):
        for c_idx in range(0,c):
            i = r_idx *c + c_idx
            array[r_idx*row_size:(r_idx+1)*row_size, c_idx*col_size: (c_idx+1)*col_size, :] = cmap[i]
            x = c_idx*col_size + delta
            y = r_idx*row_size + row_size/2
            s = labels[i]
            plt.text(x, y,s, fontsize=9, color='white')
            print "write {} at pixel (r={},c={})".format(labels[i], y,x)

    array[r*row_size:(r+1)*row_size, :] = cmap[-1]
    x = 3*col_size + delta
    y = r*row_size + row_size/2
    s = labels[-1]
    plt.text(x, y,s, fontsize=9, color='black')
    print "write {} at pixel (r={},c={})".format(labels[i], y,x)
    plt.title("PASCAL VOC Label Color Map")
    imshow(array)
    axis = plt.subplot(1, 1, 1)
    plt.axis('off')
    if fname:
        plt.savefig(fname, dpi=300,bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()

def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,'*.' + FLAGS.segmentation_format))
  
  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    
    """ added by CCJ """
    if 0:
        src_annota = cv2.imread(annotation)
        print "annotations = ", annotations
        #cv2.imshow('src_annota',src_annota)
        #cv2.waitKey()
        palette = Image.open(annotation).getpalette()
        color_map_info(palette)
        color_map_info(np.reshape(color_map(), [-1,]))
        color_map_viz(fname = "/home/ccj/seg-depth/datasets/pascal_voc_seg/pascal-voc-label-color-map.jpg")
        #cv2.imshow('raw_annota',raw_annotation)
        #cv2.waitKey()
        print type(src_annota), type(raw_annotation)
        print type(src_annota[151,160, 0]), type(raw_annotation[151,160])
        print 'src pixel (151, 160) = {}, raw pixel (151, 160) = {}'.format(src_annota[151,160], raw_annotation[151,160])
    
    filename = os.path.splitext(os.path.basename(annotation))[0]
    print "filename = ", filename + '.' + FLAGS.segmentation_format
    _save_annotation(raw_annotation,os.path.join(FLAGS.output_dir,filename + '.' + FLAGS.segmentation_format))
    #sys.exit()

if __name__ == '__main__':
  tf.app.run()
