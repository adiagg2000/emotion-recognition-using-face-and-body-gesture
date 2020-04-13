# Realtime Emotion Analysis from facial Expressions

Emotion recognition is a technique used in software that allows a program to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he/she is feeling and not just that but also showing the probabilities of mixed emotions a face could has.
The model used achieved an accuracy of 66% on the test data. The realtime analyzer assigns a suitable emoji for the current emotion.

Model implementation was done in keras.<br>

## DataSet
- Dataset used was from Kaggle fer2013 Challenge (Challenges in Representation Learning: Facial Expression Recognition Challenge).

## Model Architecture
Model was divided into base and 4 module:

Base has the following architecture:
It accepts Input image followed by a convolutional layer and Batch Normalization and ReLu Activation function and then the next layer also has a convolutional layer and batchNormalization and Relu activation.

Module has the following architecture:
It has a convolutional layer followed by BatchNormalization and then the SeperableConv layer followed by BatchNormalization and Relu activation and the next layer has SepearableConv layer followed by BatchNormalization and MaxPooling layer.
  
## How to run
There are two options:
1. Realtime emotion detection, for this run:<br>
```python main.py emo_realtime```<br>
2. Emotion detection using image path, for this run:<br>
```python main.py emo_path --path <image path>```
  <br>e.g: ```python main.py emo_path --path saved_images/im.jpg```
3. Emotion detection using video, for this run:<br>
```python main.py emo_video --path <image path>```<br>
  <br><br>You can just use this with the provided pretrained model i have included in the path written in the code file, if you want to train the model, run the command:<br>
```python train_emotion_classifier.py```<br>
