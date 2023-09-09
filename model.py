from keras.models import  Model
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda, LeakyReLU, Concatenate, Add
from keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
import json


#Load configuration variables
with open("config.json","r") as f:
  data = f.read()

config = json.loads(data)


IMAGE_H = config["IMAGE_H"]
IMAGE_W = config["IMAGE_W"]
IMAGE_C = config["IMAGE_C"]
GRID_SIZE = config["GRID_SIZE"]
N_ANCHORS = config["N_ANCHORS"]
ANCHORS = config["ANCHORS"]



def full_layer(input, filters, kernel_size, pool=False) :
  out = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1),padding='same', use_bias=False,  kernel_regularizer=l2(5e-4))(input)
  out = BatchNormalization()(out)
  out = LeakyReLU(alpha=0.1)(out)
  if pool:
    out = MaxPooling2D(pool_size=(2, 2))(out)
  return out

@keras.saving.register_keras_serializable(package="CustomActivation")
class CustomActivation(keras.layers.Layer):

    def __init__(self, GRID_SIZE, N_BOX, ANCHORS, **kwargs ):
        super(CustomActivation, self).__init__(**kwargs)
        self.GRID_SIZE = GRID_SIZE
        self.N_BOX = N_BOX
        self.ANCHORS =ANCHORS

    def get_config(self):

        config = super().get_config()
        config.update({
            'GRID_SIZE': self.GRID_SIZE,
            'N_BOX': self.N_BOX,
            'ANCHORS': self.ANCHORS
        })
        return config

    def call(self, inputs):

      cell_x = tf.cast(tf.reshape(tf.tile(tf.range(self.GRID_SIZE), [self.GRID_SIZE]), (1, self.GRID_SIZE, self.GRID_SIZE, 1, 1)),dtype=float)
      cell_y = tf.transpose(cell_x, (0,2,1,3,4))
      cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [1, 1, 1, self.N_BOX, 1])

      coordinates_xy = (tf.sigmoid(inputs[..., :2]) + cell_grid) #/ 13
      coordinates_hw = tf.exp(inputs[..., 2:4]) * tf.cast(tf.reshape(self.ANCHORS,[1,1,1,self.N_BOX,2]),dtype=float)
      confidence = tf.sigmoid(inputs[..., 4:5])
      output = tf.concat([coordinates_xy, coordinates_hw, confidence], axis=-1)

      return output



def def_model(IMAGE_H ,IMAGE_W ,IMAGE_C ,GRID_SIZE ,N_ANCHORS,ANCHORS):

    input_image = Input(shape=(IMAGE_H, IMAGE_W, IMAGE_C),name="input_image")

    #BLOCK 1
    x = full_layer(input_image, 32, (3,3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #BLOCK 2
    x = full_layer(x, 64, (3,3))
    x_re = MaxPooling2D(pool_size=(2, 2))(x)

    #BLOCK 3
    x_re = full_layer(input=x_re, filters=64, kernel_size=(3,3))
    x = full_layer(input=x_re, filters=64, kernel_size=(1,1))
    x = full_layer(input=x, filters=64, kernel_size=(3,3))

    x = Add()([x,x_re])

    x_re = MaxPooling2D(pool_size=(2, 2))(x)


    #BLOCK 4
    x_re = full_layer(input=x_re, filters=128, kernel_size=(3,3))
    x = full_layer(input=x_re, filters=128, kernel_size=(1,1))
    x = full_layer(input=x, filters=128, kernel_size=(3,3))

    x = Add()([x,x_re])

    x_re = MaxPooling2D(pool_size=(2, 2))(x)

    #BLOCK 5
    x_re = full_layer(input=x_re, filters=256, kernel_size=(3,3))
    x = full_layer(input=x_re, filters=256, kernel_size=(1,1))
    x = full_layer(input=x, filters=256, kernel_size=(3,3))
    x = full_layer(input=x, filters=256, kernel_size=(1,1))
    x = full_layer(input=x, filters=256, kernel_size=(3,3))

    skip_connection = x

    x = Add()([x,x_re])

    x_re = MaxPooling2D(pool_size=(2, 2))(x)

    #BLOCK 6
    x_re = full_layer(input=x_re, filters=512, kernel_size=(3,3))
    x = full_layer(input=x_re, filters=512, kernel_size=(1,1))
    x = full_layer(input=x, filters=512, kernel_size=(3,3))
    x = full_layer(input=x, filters=512, kernel_size=(1,1))
    x = full_layer(input=x, filters=512, kernel_size=(3,3))
    x = full_layer(input=x, filters=512, kernel_size=(3,3))
    x = full_layer(input=x, filters=512, kernel_size=(3,3))

    x = Add()([x,x_re])

    skip_connection = full_layer(input=skip_connection,filters=64, kernel_size=(1,1))

    #reshape keeping spatial information
    skip_connection = Lambda(lambda skip_connection: tf.nn.space_to_depth(skip_connection, block_size=2))(skip_connection)

    x = Concatenate()([skip_connection, x])

    x = full_layer(input=x, filters=1024, kernel_size=(3,3))

    x = Conv2D(N_ANCHORS * 5, (1,1), strides=(1,1), padding='same')(x)
    output = Reshape((GRID_SIZE, GRID_SIZE, N_ANCHORS, 5),name="final_output")(x)
    output = CustomActivation(GRID_SIZE, N_ANCHORS, ANCHORS  )(output)

    model = Model(input_image , output)
    
    return model


model = def_model(IMAGE_H ,IMAGE_W ,IMAGE_C ,GRID_SIZE ,N_ANCHORS,ANCHORS)

#model.summary()