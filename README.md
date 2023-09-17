# Vehicle-license-plate-recognition
## Description
The project aims to detect vehicle's license plates present in images in real world scenarios,
and recognize the characters on them.
In order to do so, two convolutional neural networks are used: the first for detecting the
plates and the second for recognizing the characters on them.
The tools used for the models are tensorflow and keras, for an easy and effective design,
with the help of openCV and numpy for data processing.
## Vehicle's plate detection
The model for vehicle plate detection is defined in `model_CPD.py`. It is a single stage
object detection model, based on the Yolo architecture. The model is 22 layers deep and it
is basically a sequence of stacks of convolutions with Leaky Relu activations and max
pooling at the end of each stack. It uses Batch Normalization for better stability during
training and for preventing overfitting. The model uses also
residual connections in order to avoid zeroing the gradient and exploiting the mixing of
different level features. Other techniques used to avoid overfitting are early stopping, data
augmentation, and L2-regularization.

For training the input images are divided into 13x13 grid cells. The cells that contain the
center of a plate will contain the coordinates of the bounding boxes containing them. 5
anchors boxes are used to better shape the output with respect to
the images in the dataset. The output will be a 13x13x5x5 tensor: for each cell and each
anchor, will be predicted a bounding box shaped like this:

`Bounding box <x> <y> <w> <h> <c>`

where `x` and `y` are the coordinates of the center of the box, `w` and `h` are the dimensions
of the box and `c` is the confidence of the box that measures how confident is the model of
the prediction.

The loss function is a combination of a MSE on the box coordinates and on the confidence
value, where the ground truth is the Intersection over Union between the predicted box and
the ground truth box.

For prediction outputs are thresholded on the confidence value and a non maxima
suppression is computed in order to avoid multiple predictions of the same object.

## Character recognition
The second model is a simpler CNN that receives in input the image of a character and
predicts its value (class). It is a 6 layer's deep model which uses 2D convolutions max
pooling, and a dense final layer to predict the one-hot encoded vector for classification.
There are 36 possible classes (A to Z and 0 to 9). It also makes use of dropout technique.

Input data are also augmented.

The loss function of the model is categorical cross-entropy
## Full pipeline
The full pipeline of a prediction is:

1) Load an image and preprocess it

2) Predict the location of the plates in the image, using the first model

3) For each predicted plate compute segment each character, using openCV functions

4) Each image of a character is passed to the second model which will predict its value, and
the whole word will be formed by the their sequence

The final result will be the location and the number (the sequence of numbers and
characters) of the plate, which can after be shown on the initial image
## Evaluation
The first model reach an Average Precision of 0.71, computed using a threshold over the
IoU of 0.5 between the predictions and their ground truths, meaning that each predictions
that has a IoU greater or equal of 0.5 with the ground truth will be classified as True Positive,
otherwise it will be classified ad False Positive. Tee second model reached a precision of
0.93 for class prediction.
