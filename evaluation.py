import tensorflow as tf
import numpy as np
import json
from loss import ious
from data_processing import make_file_list, My_Custom_Generator

#Load configuration parameters
with open("config.json","r") as f:
  data = f.read()

config = json.loads(data)


def treshold_output(prediction,th1=0.5,th2=0.01):
  """ 
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

def get_true_labels(labels):
  """given a list of labels in the from (13x13x5x5) it filters only the labels which are responsable for a prediction
  and return a list of them"""
  labels = np.reshape(labels,(13*13*5,5))
  return np.array([l for l in labels if l[4]==1])

def list_all_trueBoxes(test_generator):
  """
  Iterate over all the images in the test set and return a list of all labels responsable for predictions; each label will
  also contain the id of the image it is referrd to"
  """
  all_true_boxes = []

  for i in range(32):
    img_batch,lab_batch=test_generator.__getitem__(i)
    for j in range(32):
      labels = get_true_labels(lab_batch[j])
      for k in range(len(labels)):
        all_true_boxes.append([i*32+j,labels[k,0],labels[k,1],labels[k,2],labels[k,3],labels[k,4]])

  all_true_boxes=np.array(all_true_boxes)

  return all_true_boxes


def list_all_predictionsBoxes(all_prediction):
  """Execute a prediciton for all the images in the test set andreturn a list of all predictions filtered out by  
  non maxima suppression; each prediction will also contain the id of the image it is referrd to"""
  all_pred_boxes=[]
  for i in range(len(all_prediction)):
    predictions = treshold_output(all_prediction[i])
    for j in range(len(predictions)):
      if len(predictions[j])==0 : continue
      all_pred_boxes.append([i,predictions[j,0],predictions[j,1],predictions[j,2],predictions[j,3],predictions[j,4]])

  all_pred_boxes=np.array(all_pred_boxes)

  sorted_indx=np.flip(np.argsort(all_pred_boxes[...,5]))
  #boxes sorted by confidence
  all_pred_boxes = np.array([all_pred_boxes[i] for i in sorted_indx])

  return all_pred_boxes

from sklearn.metrics import average_precision_score
from collections import Counter


def precision_recall(all_true_boxes,all_pred_boxes, th=0.5):
  """Compute precision and recall over the test set for a fixed value of threshold;
  the threshold is the minimum IoU accepted so that a prediciton will be considered a True Positive"""
  TP = np.zeros((len(all_pred_boxes)))
  FP = np.zeros((len(all_pred_boxes)))

  prec = []
  rec  = []

  n_all_true_boxes=len(all_true_boxes)

  for pred_idx, pred in enumerate(all_pred_boxes):
    pred = tf.cast(pred,dtype=float)
    labels = all_true_boxes[all_true_boxes[:,0] == pred[0]]
    n_labels = len(labels)

    if n_labels==0:
      best_iou=0
    else:
      ious_ = []
      for lab in labels:
        lab = tf.cast(lab,dtype=float)
        ious_.append(ious(lab[1:5],pred[1:5]))
      ious_=np.array(ious_)
      best_iou = np.max(ious_)
      if best_iou > th:
        TP[pred_idx] = 1
      else:
        FP[pred_idx] = 1

      TOT_TP = np.sum(TP, axis=0)
      TOT_FP = np.sum(FP, axis=0)
      recalls = TOT_TP / (n_all_true_boxes + 1e-15)
      precisions = np.divide(TOT_TP, (TOT_TP + TOT_FP + 1e-15))
      rec.append(recalls)
      prec.append(precisions)

  rec=np.array(rec)
  prec=np.array(prec)

  return (prec,rec)


def compute_AveragePrecison(all_true_boxes,all_pred_boxes,ths=0.5):
  """Compute Average Precision AP  over the test set using a given value of IoU threshold"""
  (prec,rec) = precision_recall(all_true_boxes,all_pred_boxes,ths)
  AP = np.trapz(prec, rec)
  return AP

def compute_MeanAveragePrecision(all_true_boxes,all_pred_boxes):
  """Compute Mean Average Precison using different values of IoU threshold"""
  ths = [0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.99]  #could be modified
  mAP = 0
  for th in ths:
    AP = compute_AveragePrecison(all_true_boxes,all_pred_boxes,th)
    mAP += AP
    print("ths: {} \nAP: {}\n".format(th,AP))
  mAP /= len(ths)
  return mAP


test_list=make_file_list(config["TEST_PATH"])
test_generator = My_Custom_Generator(test_list, config["BATCH_SIZE"], config["GRID_SIZE"],config["ANCHORS"])


model = tf.keras.model.load(...)  # use the path in which the model has been saved
loss,coordinateLoss, confidenceLoss, average_IoU = model.evaluate(test_generator)

print("Loss: {}\nCoordinate Loss: {}\nConfidenceLoss: {}\nAverage Intersection over Union: {}".format(loss,coordinateLoss, confidenceLoss, average_IoU))




all_true_boxes = list_all_trueBoxes(test_generator)
all_predictions=model.predict(test_generator)
all_pred_boxes = list_all_predictionsBoxes(all_predictions)

precision,recall = precision_recall(all_true_boxes,all_pred_boxes)
AP  = compute_AveragePrecison(all_true_boxes,all_pred_boxes)
mAP = compute_MeanAveragePrecision(all_true_boxes,all_pred_boxes)

print("Average Precision AP (with IoU threshold=0.5): {}\nMean Average Precision over a lis of IoUthresholds: {}".format(AP,mAP))

###PLOT THE PRECISION-RECALL CURVE 
# plt.figure(figsize=(6, 6))
# plt.subplot(2, 2, (1,2))
# plt.plot(rec, prec, label='RP curve')
# plt.title('RP curve')
# plt.xlabel('RECALL')
# plt.ylabel('PRECISION')
# plt.legend()




