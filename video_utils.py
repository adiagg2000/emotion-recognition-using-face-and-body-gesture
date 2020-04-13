import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from keras.preprocessing.image import img_to_array
from prediction_utils import prediction_path

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (64, 64))
    return True

# runs the realtime emotion detection 
def video_emotions(path):
    # load keras model
    emotion_model_path = 'models/model.hdf5'
    detection_model_path = 'haarcascades/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    print('Model loaded')

    # save location for image
    save_loc = 'save_loc/1.jpg'
    # numpy matrix for stroing prediction
    result = np.array((1,7))
    # for knowing whether prediction has started or not
    once = False
    # list of given emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

    # store the emoji coreesponding to different emotions
    emoji_faces = [] 
    for index, emotion in enumerate(EMOTIONS):
        emoji_faces.append(cv2.imread('emojis/' + emotion.lower()  + '.png', -1))

    # set video capture device , webcam in this case
    video_capture = cv2.VideoCapture(path)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    # save current time
    prev_time = time.time()

    # start webcam feed
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        cv2.imwrite(save_loc,frame)
        frame = prediction_path(save_loc)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()