import os
import numpy as np
import cv2 as cv
import random
import keras



######
#create a list of filenames for the images in the directory_path
def make_file_list(directory_path):
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
                            lab.append([float(words[3]),float(words[4]),float(words[5]),float(words[6])])
                file.close()
                #for computational limits, images with more than 4 plates are discarded
                if len(lab)<=4:                
                    file_list.append(dict(obj_name=obj_name,image_path=obj_path_noEx+".jpg",label_path=obj_path_noEx+".txt",labels=lab))
    return file_list

#preprocess the image and create a 7x7x10 label tensor     
def preprocessing(img_path,labs):
    img=cv.imread(img_path)

    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0:2]
    
    #resize and normalize image
    img = cv.resize(img, (448, 448))
    img = img / 255.

    #7x7 Grid and 2 Bounding Box per cell max
    lab_t = np.zeros([7, 7, 5])

    for l in labs:
        l = np.array(l, dtype=int)
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]

        #center of bounding box (x,y) its width and height (w,h) normalized
        x = (xmin + xmax) / 2 / img_w  
        y = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        #index of the cell in which the bounding box is in
        grid_pos = [7 * x, 7 * y]
        cell_ind_i = int(grid_pos[1])
        cell_ind_j = int(grid_pos[0])


        #position in the cell of the center of bounding box
        y = grid_pos[1] - cell_ind_i
        x = grid_pos[0] - cell_ind_j


        if lab_t[cell_ind_i,cell_ind_j, 0] == 0:
            lab_t[cell_ind_i, cell_ind_j, 0] = 1
            lab_t[cell_ind_i, cell_ind_j, 1:5] = [x, y, w, h]
        

        #wrong, multiple bb in the same cell in the input should be handled with anchor boxes
        # elif lab_t[cell_ind_i,cell_ind_j, 5] == 0:
        #     lab_t[cell_ind_i, cell_ind_j, 5] = 1
        #     lab_t[cell_ind_i, cell_ind_j, 6:] = [x, y, w, h]
        
        #Since the max number of Bounding Box in each grid cell is 2, if more than 2 bounding box are in the same cell 
        #from the third on they will be discarded


    return img, lab_t



class batch_set_Generator(keras.utils.Sequence) :
  
  def __init__(self, images_paths, labels, batch_size) :
    self.images_paths = images_paths
    self.labels = labels
    self.batch_size = batch_size

  def __len__(self) :
    return (np.ceil(len(self.images_paths) / float(self.batch_size))).astype(int)
      
  def __getitem__(self, i=0) :

    batch_images = self.images_paths[i * self.batch_size : (i+1) * self.batch_size]
    batch_labels = self.labels[i * self.batch_size : (i+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_images)):
      img_path = batch_images[i]
      label = batch_labels[i]
      image, label_matrix = preprocessing(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)       



def split_dataset(file_list):

    random.shuffle(file_list)

    train_len=int(len(file_list)*0.8)
    test_len=int(len(file_list)*0.1)

    train_list=file_list[0:train_len]
    test_list=file_list[train_len : train_len+test_len]
    validation_list=file_list[train_len+test_len:]

    return train_list,validation_list,test_list










################ 
                          
                        
path_dir="/home/dario/Desktop/CarPlateDetection/OIDv4_ToolKit/OID/Dataset/full_dataset"
file_list=make_file_list(path_dir)


#Data split
train_list,validation_list,test_list=split_dataset(file_list)


train_images_paths=[i["image_path"] for i in train_list]
train_labels=[i["labels"] for i in train_list]
test_images_paths=[i["image_path"] for i in test_list]
test_labels=[i["labels"] for i in test_list]
validation_images_paths=[i["image_path"] for i in validation_list]
validation_labels=[i["labels"] for i in validation_list]


train_set=batch_set_Generator(train_images_paths,train_labels,32)
test_set=batch_set_Generator(test_images_paths,test_labels,32)
validation_set=batch_set_Generator(validation_images_paths,validation_labels,32)


#check if there are images with multiple bb in the same cell   = 68
# r=0
# for b in range(train_set.__len__()):
#   batch_images,batch_labels=train_set.__getitem__(b)
#   k=0
#   for el in batch_labels:
#       for i in range(7):
#           for j in range(7):
#               if el[i,j,5]==1:
#                 k+=1
#                 print("multiple bb detected in ",k," cell")
#   r+=k
# print(r)

batch_images,batch_labels=train_set.__getitem__(0)
print(batch_images.shape)
print(batch_labels.shape)


# cv.imshow('prova', batch_images[0])
# cv.waitKey(0)
# cv.destroyAllWindows()



# plt.imshow(batch_images[0])
# plt.waitforbuttonpress()



##SAVE
# with open("test_images", 'wb') as file:
#     pickle.dump(test_images, file)

# with open("test_labels", 'wb') as file:
#     pickle.dump(test_labels, file)

# with open("train_images", 'wb') as file:
#     pickle.dump(train_images, file)

# with open("train_labels", 'wb') as file:
#     pickle.dump(train_labels, file)
#





# # {'obj_name': 'dbc6865985607a2b', 
# 'image_path': '/home/dario/Desktop/CarPlateDetection/OIDv4_ToolKit/OID/Dataset/full_dataset/dbc6865985607a2b.jpg', 
# 'label_path': '/home/dario/Desktop/CarPlateDetection/OIDv4_ToolKit/OID/Dataset/full_dataset/dbc6865985607a2b.txt', 
# 'labels': [[586.24, 285.81982400000004, 670.72, 350.546212], [92.16, 262.108308, 126.08, 280.693296]]}





######




##load
# with open('test_images', 'rb') as file:
#     test_images = pickle.load(file)

# with open('test_labels', 'rb') as file:
#     test_labels = pickle.load(file)

# with open('train_images', 'rb') as file:
#     train_images = pickle.load(file)

# with open('train_labels', 'rb') as file:
#     train_labels = pickle.load(file)



# plt.imshow(img_res)
# plt.waitforbuttonpress()

# plt.imshow(cv.resize(create_mask(bb, img), (sz, sz)))
# plt.waitforbuttonpress()
