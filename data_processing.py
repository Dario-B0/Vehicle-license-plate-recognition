import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras



def simple_iou(box_A, box_B):
    '''
    simple version of intersection over union since both boxes top left corner are in the same point
    '''
    x = np.minimum(box_B[:, 0], box_A[0])
    y = np.minimum(box_B[:, 1], box_A[1])

    intersection = x * y
    box_A_area = box_A[0] * box_A[1]
    box_B_area = box_B[:, 0] * box_B[:, 1]

    union = box_A_area + box_B_area - intersection

    iou_ = intersection / union

    return iou_


def kmeans(boxes, k, max_iter=100, seed=2):
    """
    Compute k-means for the elements in "boxes"
    k = number of clusters
    """
    n_elements = boxes.shape[0]

    distances     = np.empty((n_elements, k))
    previous_clusters = np.zeros((n_elements,))

    np.random.seed(seed)

    # centroids are initialized randomly
    centroids = boxes[np.random.choice(n_elements, k, replace=False)]

    for iter in range(max_iter):
        #for each element in the list find the closest centorid (i.e. the belonging cluster)
        for i in range(k):

            distances[:,i] = 1 - simple_iou(centroids[i], boxes)

        closest_clusters = np.argmin(distances, axis=1)

        if (previous_clusters == closest_clusters).all():
            return centroids

        # Compute the new centroids as mean value of each cluster
        for j in range(k):               #fancy indexing
            centroids[j] = np.mean(boxes[closest_clusters == j], axis=0)

        previous_clusters = closest_clusters

    return centroids




def make_file_list(directory_path):
    """
    Produce a list in which each element is a dictionary:\\
    {\\
    obj_name = name of the file without extensions \\
    image_path = full name of the file containing the image \\
    label_path = full name of the file containing the labels associated to the image \\
    labels = list of labels associated to the in the image\\
    }    
    """
    file_list=[]
    for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                obj_name=os.path.splitext(filename)[0]
                obj_path_noEx=os.path.join(directory_path, obj_name)
                f=obj_path_noEx+".txt"
                lab=[]
                with open(f,'r') as file:
                    for line in file:
                            words=line.split()
                            lab.append([float(words[1]),float(words[2]),float(words[3]),float(words[4])])
                file.close()
                #for computational limits, images with more than 4 plates are discarded
                #if len(lab)==1:
                file_list.append(dict(obj_name=obj_name,image_path=obj_path_noEx+".jpg",label_path=obj_path_noEx+".txt",labels=lab))
    return file_list



class ImageReader(object):
    """
    Class that load each image and preprocess it
    """
    def __init__(self,size_H,size_W):

      self.size_H=size_H
      self.size_W=size_W


    def resize_img(self,img):
      """
      Load image, convert to gray scale, resize between (self.size_H, self.size_W)=(416,416) and normalize pixel values in [0,1]
      """

      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   #GRAY SCALE IMAGES

      #resize and normalize image
      img = cv.resize(img, (self.size_H, self.size_W))
      img = img / 255.

      return img

    def encode_input(self,el_file_list):

      """
      Preprocess image => resize_img(self,img)
      and preprocess accordingly the corresponding labels
      """

      img_path = el_file_list["image_path"]
      img = cv.imread(img_path)
      h, w, c = img.shape

      img = self.resize_img(img)

      labs=[]

      #lebels values are resheped between [0, GRID_SIZE] (= [0,13] )
      for l in el_file_list["labels"]:
        x = l[0] * 13
        y = l[1] * 13
        w = l[2] * 13
        h = l[3] * 13

        labs.append([x,y,w,h])

      return img,labs
    


def best_Anchor(anchors, bb_w, bb_h):
  """
  Compute the anchor that best fit the box (highest IoU)
  """

  ious=simple_iou([bb_w,bb_h],anchors)

  best_anch = tf.argmax(ious)
  best_iou = tf.reduce_max(ious)

  return best_anch, best_iou



class My_Custom_Generator(keras.utils.Sequence) :

  def __init__(self, file_list, batch_size, grid_size, anchors) :
    self.file_list = file_list
    self.batch_size = batch_size
    self.grid_size = grid_size
    self.IR = ImageReader(size_W=416,size_H=416)
    self.anchors = anchors


  def __len__(self) :
    return (np.ceil(len(self.file_list) / float(self.batch_size))).astype(int)


  def __getitem__(self, idx) :
    start_idx = idx * self.batch_size
    end_idx = (idx+1) * self.batch_size

    if end_idx > len(self.file_list):
      end_idx = len(self.file_list)
      start_idx = end_idx - self.batch_size

    batch_img = np.zeros((self.batch_size, 416, 416))
    batch_lab = np.zeros((self.batch_size, self.grid_size,  self.grid_size, self.anchors.shape[0], 5))

    for indx_el in range(self.batch_size):
        img , labs = self.IR.encode_input(self.file_list[indx_el + start_idx])

      
        for l in labs:

            cell_x = int(l[0])
            cell_y = int(l[1])

            best_anchor,max_iou = best_Anchor(self.anchors, (l[2]), (l[3]) )

            batch_lab[indx_el, cell_y, cell_x, best_anchor, 0:4] = l  # [x, y, w, h]
            batch_lab[indx_el, cell_y, cell_x, best_anchor, 4  ] = 1. 

        batch_img[indx_el] = img


    return batch_img, batch_lab