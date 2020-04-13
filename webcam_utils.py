import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from keras.preprocessing.image import img_to_array

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (64, 64))
    return True

# runs the realtime emotion detection 
def realtime_emotions():
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
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    # save current time
    prev_time = time.time()

    # start webcam feed
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = face_detection.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # required region for the face
            roi_color = gray[y:y+h, x:x+w]
            # save the detected face
            cv2.imwrite(save_loc, roi_color)
            
            # keeps track of waiting time for emotion recognition
            curr_time = time.time()
            # do prediction only when the required elapsed time has passed 
            if curr_time - prev_time >=1:
                # read the saved image
                img = cv2.imread(save_loc, 0)
                if img is not None:
                    # indicates that prediction has been done atleast once
                    once = True
                    # resize image for the model
                    img = cv2.resize(img, (64, 64))
                    img = img.astype("float") / 255.0
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    # do prediction
                    preds = emotion_classifier.predict(img)[0]
                    result = emotion_classifier.predict(img)
                    emotion_probability = np.max(preds)
                    label = EMOTIONS[preds.argmax()]
                    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                         # construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    print(label)
                    
                #save the time when the last face recognition task was done
                prev_time = time.time()

            if once == True:
                total_sum = np.sum(result[0])
                # select the emoji face with highest confidence
                emoji_face = emoji_faces[np.argmax(result[0])]
                for index, emotion in enumerate(EMOTIONS):
                    text = str(
                        round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                    # for drawing progress bar
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                    # for putting emotion labels
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    # for putting percentage confidence
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    
                # overlay emoji on the frame for all the channels
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
