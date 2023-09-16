import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #GRAY SCALE IMAGES

      #resize and normalize image
      img = cv2.resize(img, (self.size_H, self.size_W))
      img = img / 255.

      return img

    def encode_input(self,el_file_list):

      """
      Preprocess image => resize_img(self,img)
      and preprocess accordingly the corresponding labels
      """

      img_path = el_file_list["image_path"]
      img = cv2.imread(img_path)
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
  


def apply_data_augmentation(image, bbox, visualize=False):
    """Apply data augmentation to the image in input , reshaping the bounding box(es) accordingly.
       Visualize all steps of data augmentation with the "visualize" parameter """
    bbox2=bbox
    image2=image

    # Apply rotation between -45° and +45°
    angle = np.random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((image2.shape[1] // 2, image2.shape[0] // 2), angle, 1)
    image3 = cv2.warpAffine(image2, M, (image2.shape[1], image2.shape[0]),borderMode =cv2.BORDER_REPLICATE  )

    bbox_new=np.zeros((len(bbox),2,2))
    for i in range(len(bbox)):
      corners = np.array([ [bbox2[i,0],bbox2[i,1]],
                        [bbox2[i,2],bbox2[i,1]],
                        [bbox2[i,0],bbox2[i,3]],
                        [bbox2[i,2],bbox2[i,3]]],dtype=float)

      bbox_ = cv2.transform(np.array([corners]), M)[0]

      bbox_new[i,0,0] = int(np.min(bbox_[:, 0]))
      bbox_new[i,1,0] = int(np.max(bbox_[:, 0]))
      bbox_new[i,0,1] = int(np.min(bbox_[:, 1]))
      bbox_new[i,1,1] = int(np.max(bbox_[:, 1]))

    bbox3 = bbox_new.copy()

    # Apply shear between -5° and +5°
    shear_angle = np.random.uniform(-5, 5)
    M_shear = np.array([[1, np.tan(np.radians(shear_angle)), 0],
                        [0, 1, 0]])
    image4 = cv2.warpAffine(image3, M_shear, (image3.shape[1], image3.shape[0]),borderMode =cv2.BORDER_REPLICATE)

    bbox4 = bbox3.copy()
    for i in range(len(bbox)):
      bbox4[i] = cv2.transform(np.array([bbox4[i]]), M_shear).squeeze()


    # Apply hue adjustment between -15° and +15°
    hue_shift = np.random.uniform(-15, 15)
    hsv_image = cv2.cvtColor(image4, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    image5 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Apply saturation between -15% and +15%
    hsv_image = cv2.cvtColor(image5, cv2.COLOR_BGR2HSV)
    saturation_factor = np.random.uniform(0.85, 1.15)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    image6 = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Apply brightness between -30% and +30%
    brightness_factor = np.random.uniform(0.80, 1.20)
    image7 = image6 * brightness_factor
    image7 = np.clip(image7, 0, 255).astype(np.uint8)

    # Apply Gaussian blur (5,5) up to 1 sigma
    blur_sigma = np.random.uniform(0, 1)
    image8 = cv2.GaussianBlur(image7, (5, 5), blur_sigma)

    if visualize:
      fig,((ax1, ax2), (ax3, ax4),(ax5, ax6), (ax7, ax8))=plt.subplots(4,2, figsize=(12,12))
      ax1.set_axis_off()
      ax1.set_title("Original image")
      ax1.imshow(image2)

      ax2.set_axis_off()
      ax2.set_title("Rotation: +/- 45°")
      ax2.imshow(image3)

      ax3.set_axis_off()
      ax3.set_title("Shear: +/- 5°")
      ax3.imshow(image4)

      ax4.set_axis_off()
      ax4.set_title("Hue: +/- 15°")
      ax4.imshow(image5)

      ax5.set_axis_off()
      ax5.set_title("Saturation: +/- 15%")
      ax5.imshow(image6)

      ax6.set_axis_off()
      ax6.set_title("Brightness: +/- 30%")
      ax6.imshow(image7)

      ax7.set_axis_off()
      ax7.set_title("Gausian blur (5,5): 0 <= simga <= 2")
      ax7.imshow(image8)

      for i in range(len(bbox)):

        rect = patches.Rectangle((bbox2[i,0], bbox2[i,1]), ( bbox2[i,2]-bbox2[i,0] ), ( bbox2[i,3]-bbox2[i,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

        rect = patches.Rectangle((bbox_new[i,0,0], bbox_new[i,0,1]), ( bbox_new[i,1,0]-bbox_new[i,0,0] ), ( bbox_new[i,1,1]-bbox_new[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

        rect = patches.Rectangle((bbox4[i,0,0], bbox4[i,0,1]), ( bbox4[i,1,0]-bbox4[i,0,0] ), ( bbox4[i,1,1]-bbox4[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)

        rect = patches.Rectangle((bbox4[i,0,0], bbox4[i,0,1]), ( bbox4[i,1,0]-bbox4[i,0,0] ), ( bbox4[i,1,1]-bbox4[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax4.add_patch(rect)
        rect = patches.Rectangle((bbox4[i,0,0], bbox4[i,0,1]), ( bbox4[i,1,0]-bbox4[i,0,0] ), ( bbox4[i,1,1]-bbox4[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax5.add_patch(rect)
        rect = patches.Rectangle((bbox4[i,0,0], bbox4[i,0,1]), ( bbox4[i,1,0]-bbox4[i,0,0] ), ( bbox4[i,1,1]-bbox4[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax6.add_patch(rect)
        rect = patches.Rectangle((bbox4[i,0,0], bbox4[i,0,1]), ( bbox4[i,1,0]-bbox4[i,0,0] ), ( bbox4[i,1,1]-bbox4[i,0,1] ), linewidth=1, edgecolor='r', facecolor='none')
        ax7.add_patch(rect)

      plt.show()

    return image8, bbox4



def dataAugmentation(train_list):
  """Apply data augmentation two times to all elements in the train list """
  for i in range(len(train_list)):
    if i%100==0:
      print(i)
    image = cv2.imread(train_list[i]["image_path"])
    image = cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),(416,416))
    bbox = np.array(train_list[i]["labels"])

    for j in range(len(bbox)):
      bbox[j] = np.array([bbox[j][0] - bbox[j][2]/2 ,bbox[j][1] - bbox[j][3]/2 ,bbox[j][0] + bbox[j][2]/2 ,bbox[j][1] + bbox[j][3]/2 ]) * 416

    aug_img1,aug_bbox1_ = apply_data_augmentation(image, bbox, visualize=False)
    aug_img2,aug_bbox2_ = apply_data_augmentation(image, bbox, visualize=False)

    name_obj = train_list[i]["obj_name"]

    aug_bbox1 = np.zeros((len(bbox), 4))
    aug_bbox2 = np.zeros((len(bbox), 4))

    cv2.imwrite("train/" +name_obj+"_AUG1.jpg", aug_img1)
    cv2.imwrite("train/" +name_obj+"AUG2.jpg", aug_img2)

    with open("train/" + name_obj + "_AUG1.txt", 'w') as f1:
      with open("train/" + name_obj +"_AUG2.txt", 'w') as f2:
        str_box1=""
        str_box2=""
        for i in range(len(bbox)):
          # xmin ymin xmax ymax ==> xc yc w h
          aug_bbox1[i,0] = (aug_bbox1_[i,0,0] + aug_bbox1_[i,1,0] ) /2
          aug_bbox2[i,0] = (aug_bbox2_[i,0,0] + aug_bbox2_[i,1,0] ) /2
          aug_bbox1[i,1] = (aug_bbox1_[i,0,1] + aug_bbox1_[i,1,1] ) /2
          aug_bbox2[i,1] = (aug_bbox2_[i,0,1] + aug_bbox2_[i,1,1] ) /2
          aug_bbox1[i,2] = (aug_bbox1_[i,0,0] + aug_bbox1_[i,1,0] ) /2
          aug_bbox2[i,2] = (aug_bbox1_[i,0,0] + aug_bbox1_[i,1,0] ) /2
          aug_bbox1[i,3] = (aug_bbox1_[i,0,0] + aug_bbox1_[i,1,0] ) /2
          aug_bbox2[i,3] = (aug_bbox1_[i,0,0] + aug_bbox1_[i,1,0] ) /2

          aug_bbox1 = aug_bbox1 / 416
          aug_bbox2 = aug_bbox2 / 416

          str_box1 = str_box1 + "0 {} {} {} {}\n".format(aug_bbox1[i][0], aug_bbox1[i][1], aug_bbox1[i][2], aug_bbox1[i][3])
          str_box2 = str_box2 + "0 {} {} {} {}\n".format(aug_bbox2[i][0], aug_bbox2[i][1], aug_bbox2[i][2], aug_bbox2[i][3])
        str_box1 = str_box1.rstrip()
        str_box2 = str_box2.rstrip()
        f1.write(str_box1)
        f2.write(str_box2)

  return

def perform_data_augmentation(train_path):
  """
  This function perform the data augmentation on the dataset; 
  it should be called only once before training
  """
  train_list=make_file_list(train_path)
  dataAugmentation(train_list)
