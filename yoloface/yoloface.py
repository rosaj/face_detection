# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm (https://github.com/sthanhng/yoloface)
#
#
# The main code of the Face detection using the YOLOv3 algorithm
# It is a YOLOv3 model trained on http://shuoyang1213.me/WIDERFACE/index.html dataset
#
# *******************************************************************


from .utils import *

Name = 'YOLOFace'


def __load_model():
    # Give the configuration and weight files for the model and load the network using them.
    net = cv2.dnn.readNetFromDarknet('./yoloface/cfg/yolov3-face.cfg',
                                     './yoloface/model-weights/yolov3-wider_16000.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


__model = __load_model()


def detect_faces(frame, conf_threshold=0.00, nms_threshold=0.00):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (1920, 1920),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    __model.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = __model.forward(get_outputs_names(__model))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, conf_threshold, nms_threshold)

    return faces
