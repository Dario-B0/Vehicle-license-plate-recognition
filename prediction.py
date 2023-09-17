import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loss_CPD import ious

#Load configuration parameters
with open("config.json","r") as f:
  data = f.read()

config = json.loads(data)

def obtain_roi_plate(image, coordinates):
    """given the image and the coordinates of a boundig box, it return a crop of the bounding box area on the image"""
    # Clona l'immagine per evitare di modificarla direttamente
    result_image = image.copy()

    # Estrai le coordinate della targa normalizzate (x, y, larghezza, altezza)
    x_norm, y_norm, width_norm, height_norm = coordinates

    # Ottieni le dimensioni dell'immagine
    height, width, _ = image.shape

    # Converti le coordinate normalizzate in coordinate in pixel
    x = int(x_norm * width)
    y = int(y_norm * height)
    width = int(width_norm * width)
    height = int(height_norm * height)

    # Calcola il punto iniziale (top-left) e il punto finale (bottom-right) del rettangolo
    start_p = (x, y)
    end_p = (x + width, y + height)

    # Disegna un rettangolo attorno alla targa
    cv2.rectangle(result_image, start_p, end_p, (0, 255, 0), 2)

    return result_image

def treshold_output(prediction,th1=0.5,th2=0.01):
  """ 
    Threshold the output of the first model, based on the confidence.
    It also removes ovelapping boxes(only the ones wit lower confidence) to avoid detecting multiple time the same plate 
    prediction (N,13,13,5,5)
    th1 = 0.5   minimum confidence required
    th2 = 0.01  max overlapping accepted
    """

  boxes = []
  for i in range(config["GRID_SIZE"]):
    for j in range(config["GRID_SIZE"]):
      for b in range(config["N_ANCHORS"]):
        if prediction[i,j,b,4] > th1:
          boxes.append(prediction[i,j,b])

  if not boxes: return []
  boxes = np.array(boxes)

  sorted_indx=np.flip(np.argsort(boxes[...,4]))
  #boxes sorted by confidence
  boxes_sorted = np.array([boxes[i] for i in sorted_indx])

  for i in range(len(boxes)):
    if boxes_sorted[i,4]==0:
      continue
    else:
      for j in range(i+1,len(boxes)):
        box_iou = ious(boxes_sorted[i,:4],boxes_sorted[j,:4])
        if box_iou >= th2:
          boxes_sorted[j,4] = 0

  boxes = [boxes_sorted[i] for i in range(len(boxes_sorted)) if boxes_sorted[i,4]!=0]

  return np.array(boxes)


# Find characters
def segmentation(image) :
    """Execute the segmentation on the image and retrive only the characters"""

    # Preprocess on the cropped license plate image
    img_gray = cv2.cvtColor(cv2.resize(image, (333, 75)),cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary = cv2.dilatate(cv2.erode(img_binary, (3,3)),(3,3))

    bounds = [5,50,30,50]

    #char_list = find_contours(dimensions, img_binary)

    # Find contours in the image
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:16]

    plate_binary = cv2.imread('plate_binary.jpg')

    x_list = []
    char_list = []
    for contour in contours :
        # detects contour in binary image and returns the coordinates of rectangle containing it
        x, y, w, h = cv2.boundingRect(contour)

        # take only contours fitting bounds
        if w > bounds[0] and w < bounds[1] and h > bounds[2] and h < bounds[3] :
            x_list.append(x) #used for sorting

            char_res = np.zeros((44,24))
            char = img_binary[y:y+h, x:x+w]
            char = cv2.resize(char, (20, 40))
            cv2.rectangle(plate_binary, (x,y), (w+x, y+h), (50,21,200), 2)

            # trasnform for the network
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_res[2:42, 2:22] = char
            char_res[0:2, :] = 0
            char_res[:, 0:2] = 0
            char_res[42:44, :] = 0
            char_res[:, 22:24] = 0

            char_list.append(char_res) 

    # Sort the characters by position
    x_list=np.array(x_list)
    idxs = np.argsort(x_list)
    img_res_copy = []
    for idx in idxs:
        img_res_copy.append(char_list[idx])  
    char_list = np.array(img_res_copy)

    return char_list

def fix_dimension(img):
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img

def show_results(char,model):
    """Given a list of images of a characters and the model, it predicts the whole word"""
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): 
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) 
        y_ = model.predict(img) 
        character = characters[np.argmax(y_)] 
        output.append(character) 

    plate_number = ''.join(output)

    return plate_number

def full_prediction(img_path, model_cpl, model_ocr):
  """Execute the full pipeline:
  -Load an image of a car in real word scenario and preprocess it
  -Execute prediciton on the first model to detect the location of the plate
  -For each plate:
    -Segment the characters using opencv function
    -Exectute prediction on each caracter using the second model and retrive the plate number
  The final result will be the initial image with the prediction of the location of the plates and the numbers on them"""
  img=cv2.imread(img_path)

  fig, ax = plt.subplots()

  ax.set_axis_off()
  ax.imshow(img)

  img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #GRAY SCALE IMAGES
  img_ = cv2.resize(img, (416, 416))
  img_ = img / 255.

  img_= np.array(img)
  img_= np.reshape(img_, (1,416,416))
  pred= model_cpl(img_)

  thresh_pred = treshold_output(pred[0])

  sorted_indx=np.flip(np.argsort(thresh_pred[...,4]))
  #boxes sorted by confidence
  pred = np.array([thresh_pred[i] for i in sorted_indx])
  for i in range(len(pred)):
    p=(pred[i][0:4]/13)
    p[0]=p[0]-p[2]/2  #since it returns the central pixel
    p[1]=p[1]-p[3]/2

    plate_img = obtain_roi_plate(img[0], p)

    cv2.imwrite("plate_region.jpg",plate_img*255)

    plate_img=cv2.imread("plate_region.jpg")

    char=segmentation(plate_img)

    plate_number = show_results(char,model_ocr)

    p=p*416
    ax.add_patch(patches.Rectangle((p[0], p[1]), p[2], p[3], linewidth=3, edgecolor='r', facecolor='none'))
    ax.text(p[0] -2, p[1] - 11, '{}'.format(plate_number), fontsize = 10, color="r", fontweight='bold')

  plt.show()