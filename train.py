import numpy as np
import json
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from loss import yoloLoss, ConfidenceLoss, CoordLoss, metric_avg_iou
from data_processing import make_file_list, kmeans, My_Custom_Generator
from model import def_model


#Load configuration parameters
with open("config.json","r") as f:
  data = f.read()

config = json.loads(data)


#List elements for each set
test_list=make_file_list(config["TEST_PATH"])
train_list=make_file_list(config["TRAIN_PATH"])
validation_list=make_file_list(config["VALID_PATH"])


#K-means in order to find the best anchors
full_list=np.concatenate(( np.array(test_list),np.array(train_list),np.array(validation_list) ), axis=None)

wh=[]
for el in full_list:
  for l in el["labels"]:
    wh.append([l[2],l[3]])

#Contains all the labels in the dataset
wh=np.array(wh)

anch = kmeans(wh,config["N_ANCHORS"],seed=2)
anch *= 13                                    #rescale in [0:13]
#config["ANCHORS"] = anch
#print(anch)
#


#Data generator (inputs of the model)
train_generator = My_Custom_Generator(train_list, config["BATCH_SIZE"], config["GRID_SIZE"],config["ANCHORS"])
validation_generator = My_Custom_Generator(validation_list, config["BATCH_SIZE"], config["GRID_SIZE"],config["ANCHORS"])
test_generator = My_Custom_Generator(test_list, config["BATCH_SIZE"], config["GRID_SIZE"],config["ANCHORS"])


#model callbacks
mcp_save = ModelCheckpoint('weight_yv2_13.hdf5', save_best_only=True, monitor='val_loss', mode='min')
decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, 6000, 0.95, staircase=True)

#load and compile model
### load model from file or
### load define model from model.py and assign it
### load model from model.py directly defined


model = def_model(config["IMAGE_H"],config["IMAGE_W"] ,config["IMAGE_C"] ,config["GRID_SIZE"] ,config["N_ANCHORS"],config["ANCHORS"])

model.compile(loss= yoloLoss, optimizer=tf.keras.optimizers.Adam(learning_rate=decayed_lr,clipnorm=.2), metrics=[CoordLoss,ConfidenceLoss,metric_avg_iou])



history_detection = model.fit(
	train_generator,
	validation_data=validation_generator,
	epochs=70,
	verbose=1,
	callbacks=[mcp_save])


#save history
with open('training_history.json', 'w') as f:
    json.dump(history_detection.history, f)