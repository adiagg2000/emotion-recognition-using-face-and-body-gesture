from keras.preprocessing.image import img_to_array
import imutils
import cv2
import os.path
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# make prediction on image saved on disk
def prediction_path(path):
    # load keras model
    emotion_model_path = 'models/model.hdf5'
    detection_model_path = 'haarcascades/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    # list of given emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
    if os.path.exists(path):
        # read the image
        frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
        
    frameClone = frame.copy()
    for (fX,fY,fW,fH) in faces: 
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        print(label)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        #frameClone = cv2.resize(frameClone, (640, 480), interpolation = cv2.INTER_LINEAR) 
        #cv2.imshow('image', frameClone)
    #imgplot = plt.imshow(frameClone)
    #plt.show()

    return frameClone
