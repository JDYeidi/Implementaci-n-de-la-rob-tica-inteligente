# Author    :   Juan Aranda, Paul Garcia
# Based on  :   Ricardo Acevedo Avila
# License   :   ITESM

# Import the necessary packages:
from code import interact
from configparser import Interpolation
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from imutils import paths
import os

# Setup video source:
video = cv2.VideoCapture(0)

# Set the resources paths:
# mainPath 
mainPath = os.path.join("/home/paul/VSCode/Python/DeepL", "traffic-sign")
# modelPath
modelPath = os.path.join(mainPath, "output")

# Training image size:
imageSize = (64, 64)

# The class dictionary:
classDictionary = {0: "Ahead only", 1: "End of all speed and passing limits", 2: "Roundabout mandatory", 3: "Stop", 4: "Turn right ahead"}

# Load model:
print("[SignClass - Test] Loading network...")
model = load_model(os.path.join(modelPath, "signclass.model"))

# Loop over the video frames and classify each one:
while True:

    # Load the frame via OpenCV:
    ret, image = video.read()

    # Deep copy for displaying results:
    output = image.copy()

    # Obteining height and width:
    height, width, _ = image.shape

    # Pre-process the image and resizing for classification:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    image = image.reshape(1, 64, 64, 3)
    image = image.astype("float") / 255.0  

    # Classify the input image
    print("[SignClass - Test] Classifying image...")

    # Send to CNN, get probabilities:
    predictions = model.predict(image)

    # Get max probability, thus, the max classification result:
    classIndex = predictions.argmax(axis=1)[0]
    # Get categorical label via the class dictionary:
    label = classDictionary[classIndex] 

    # Print the classification result:
    print("Class: " + label + " prob: " + str(predictions[0][classIndex]))

    # Build the label and draw the label on the image
    prob = "{:.2f}%".format(predictions[0][classIndex] * 100)
    label = label + " " + prob
    max_prob = np.amax(predictions)

    # Show if prob > 85 % :
    if max_prob > 0.85:

        # Draw Text:
        textColor = (155, 5, 170)
        cv2.putText(output, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)

        # Show the output image and its label & probability
        cv2.imshow("Output", output)

    # Low Prob -> No signal: 
    else: 
        print("No signal detected")
        cv2.imshow("Output", output)

    # Close window:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()