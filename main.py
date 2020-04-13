import argparse
from webcam_utils import realtime_emotions
from prediction_utils import prediction_path
from video_utils import video_emotions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# for running realtime emotion detection
def run_realtime_emotion():
    realtime_emotions()

# to run emotion detection on image saved on disk
def run_detection_path(path):
    img = prediction_path(path)
    imgplot = plt.imshow(img)
    plt.show()

def run_video_emotion(path):
    video_emotions(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func_name", type=str,
                        help="Select a function to run. <emo_realtime> or <emo_path> or <emo_video>")
    parser.add_argument("--path", type=str,help="Specify the complete path where the image/video is saved.")
    # parse the args
    args = parser.parse_args()

    #print('****ARGS: ' + str(args))

    if args.func_name == "emo_realtime":
        run_realtime_emotion()
    elif args.func_name == "emo_video":
        run_video_emotion(args.path)
    elif args.func_name == "emo_path":
        run_detection_path(args.path)
    else:
        print("Usage: python main.py <function name>")

if __name__ == '__main__':
    main()
