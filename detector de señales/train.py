# Author    :   Juan Aranda, Paul Garcia
# Based on  :   Ricardo Acevedo Avila
# License   :   ITESM

# Import the layers code:
from sign_cnn import signCNN

# Import keras modules:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Label binarizer:
from sklearn.preprocessing import LabelBinarizer

# The dataset split package:
from sklearn.model_selection import train_test_split

# Import matplot for plotting:
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib's renderer:
matplotlib.use("Agg")

# Path management:
from imutils import paths

# Other packages:
import numpy as np
import random
import pickle
import cv2
import os
from datetime import datetime


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Training early stop callback.
# The callback is called when a epoch ends:
class stopTrainCallback(Callback):
    def __init__(self, threshold, counter):
        super(stopTrainCallback, self).__init__()
        # Set the threshold value for stopping
        self.threshold = threshold
        # Set the epoch counter:
        self.epochCounter = counter

    def on_epoch_end(self, epoch, logs=None):
        # Increase the epoch counter:
        self.epochCounter += 1
        # The value I'm monitoring is the validation loss:
        accuracy = logs["val_loss"]
        # print(" Monitor: " + str(accuracy))
        # If threshold (min value) reached, end training:
        if accuracy <= self.threshold:
            global epochs
            epochs = self.epochCounter
            self.model.stop_training = True


# Set the resources paths:

# Main directory
mainPath = os.path.join("/home/paul/VSCode/Python/DeepL", "traffic-sign")
# Dataset path
dataSetPath = os.path.join(mainPath, "dataset")
# Output directory
outputPath = os.path.join(mainPath, "output")

# Training hyper parameters:
epochs = 100     # Increase epochs
gamma = 1.0     # Learning rate factor, increase in 0.1 units
learningRate = gamma * 1e-3
batchSize = 32  # Could be: 16, 32, 64 In general, the smaller the better
imageDimensions = (64, 64, 3)  # The image is resized to these dimensions

# Callback epoch counter:
epochCounter = 0

# initialize the data and labels
data = []
labels = []

# Image counter:
imageCounter = 0

# Load each image path of the dataset:
print("[SignClass - Train] Loading images...")
imagePaths = sorted(list(paths.list_images(dataSetPath)))

# Randomly shuffle the paths:
print("[SignClass - Train] Shuffling images...")

# First time...:
random.seed(datetime.now())
random.shuffle(imagePaths)

# ...and again:
random.seed(datetime.now())
random.shuffle(imagePaths)

# Loop over the input images and load em:
for imagePath in imagePaths:
    
    # Read the image via OpenCV:
    image = cv2.imread(imagePath)
    # Reorder the channels from BGR (OpenCV's default) to RGB:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to CNN dimensions:
    image = cv2.resize(image, (imageDimensions[1], imageDimensions[0]))

    # The image goes into the list:
    data.append(image)

    # Extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Convert lists to numpy arrays and scale the
# raw pixel intensities to the range [0, 1]:
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# One-hot encode the labels:
lb = LabelBinarizer()
labelsEncoded = lb.fit_transform(labels)

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labelsEncoded, test_size=0.2, random_state=42069)

print("[SignClass - Train] Images Loaded: " + str(imageCounter))
print("[SignClass - Train] Train Samples: " + str(trainX.shape[0]))
print("[SignClass - Train] Test Samples: " + str(testX.shape[0]))

# Check out the sample distribution. We can indicate the model
# which class is under represented, so it can handle it accordingly.
# This vector holds all the samples per class:
classTotal = trainY.sum(axis=0)
print("[SignClass - Train] Sample count per class: ", classTotal)

# Class weight calculation:
classWeight = classTotal.max() / classTotal

# Create a dictionary that stores the weights per class:
classWeight = {i: classWeight[i] for i in range(len(classWeight))}

# Dataset Augmentation:
# Construct the image generator for data augmentation:
aug = ImageDataGenerator(
    rotation_range=35,
    zoom_range=0.35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    horizontal_flip=True,
    fill_mode="nearest")

# Initialize the model
print("[SignClass - Train] Setting up the CNN...")

# Set up the CNN:
model = signCNN.build(width=imageDimensions[1], height=imageDimensions[0], depth=imageDimensions[2],
                      classes=len(classTotal))

# Set up the optimizer
cnnOptimizer = Adam(lr=learningRate, decay=learningRate / (epochs * 0.5))

# Compile the model:
model.compile(loss="categorical_crossentropy", optimizer=cnnOptimizer, metrics=["accuracy"])

# Train the network
print("[SignClass - Train] Training CNN...")

# Set early stop callback on validation loss with
# a threshold equal to or lower than 0.19:
callback = stopTrainCallback(threshold=0.19, counter=epochCounter)

# Actually train the network:
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batchSize),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // batchSize,
    epochs=epochs,
    class_weight=classWeight,
    verbose=1
    #callbacks=[callback]
    )

# Save the model to disk:
print("[SignClass - Train] Saving model to disk...")
# model.save
model.save(os.path.join(outputPath, "signclass.model"))

# Save the label binarizer to disk:
print("[SignClass - Train] Serializing label binarizer...")

f = open(os.path.join(outputPath, "labels.pickle"), "wb")
f.write(pickle.dumps(lb))
f.close()

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()

# Get the historical data:
N = np.arange(0, epochs)

# Plot values:
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# Save plot to disk:
plt.savefig(outputPath)
print("[SignClass - Train] Saved Loss and Accuracy Plot to: "+outputPath)
print("[SignClass - Train] Done CNN training.")