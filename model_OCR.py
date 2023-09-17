import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import   Conv2D, Input, MaxPooling2D,  Dense,  Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Model
import json

train_ds = tf.keras.utils.image_dataset_from_directory(
  '/home/dario/Desktop/VISIOPE/data/train',
  validation_split=0.2,
  subset="training",
  seed=100,
  image_size=(24, 24),
  batch_size=20)

val_ds = tf.keras.utils.image_dataset_from_directory(
  '/home/dario/Desktop/VISIOPE/data/val',
  validation_split=0.2,
  subset="validation",
  seed=100,
  image_size=(24, 24),
  batch_size=2)

x_train, y_train = np.concatenate(list(train_ds.map(lambda x, y: x))), np.concatenate(list(train_ds.map(lambda x, y: y)))

datagen = ImageDataGenerator(
    rescale = 1. / 255,
    zoom_range=0.2,
    rotation_range=45,
    brightness_range=(0.2, 0.8),
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)



train_generator = datagen.flow_from_directory(
        '/home/dario/Desktop/VISIOPE/data/train',
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        '/home/dario/Desktop/VISIOPE/data/train',
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='categorical')



def Recogn(input_shape, num_classes=36,  lr=0.0001): 

    input_image = Input(shape=(input_shape),name="input_image")

    x = Conv2D(filters=64, kernel_size=(3, 3) ,padding='same')(input_image)
    x = Conv2D(filters=16, kernel_size=(22, 22) ,padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(16, 16) ,padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(8, 8) ,padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(4, 4) ,padding='same')(x)

    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(input_image , output)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

    return model



#create the model
model = Recogn(input_shape=(28, 28, 3),num_classes=36)
model.summary()

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,baseline=None, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='weight_fc.hdf5'),
]

batch_size = 1
history= model.fit(train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator,
      validation_steps = validation_generator.samples // batch_size,
      callbacks=my_callbacks ,
      epochs = 50)


with open('training_history_recogn.json', 'w') as f:
    json.dump(history.history, f)

model.save("model_OCR.h5")
