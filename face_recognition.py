# import libraries
from utils import *
from config import *
from os import listdir
from os.path import isfile, join
import numpy as np

# list the training images by checking if the path is valid
onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]

# create empty arrays for training data and labels
Training_Data, Labels = [], []

# load the training images in the empty arrays
for i, files in enumerate(onlyfiles):
    image_path = DATA_PATH + onlyfiles[i]  # face/user1.jpg
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# convert labels array into a numpy array
Labels = np.asarray(Labels, dtype=np.int32)

# load the Linear Binary Phase Histogram Classifier
model = cv2.face.LBPHFaceRecognizer_create()
""" if this line generates error, run the following command

python -m pip install --user opencv-contrib-python

"""

# train the model
model.train(np.asarray(Training_Data), np.asarray(Labels))

print('Model Training Complete !!!')

# code to use the facial recognition using opencv
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + '% Confidence it is USER'

        cv2.putText(image, display_string, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 55, 255), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
