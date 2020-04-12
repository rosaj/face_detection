# Collection of face detection algorithms and One-Shot Face recognition



## Face detection
\

### Detectors

- [CenterFace](https://github.com/Star-Clouds/CenterFace)
  - Model (7.3Mb) already included in this repository
- [DSFD](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)
  - Implementation from [this repository](https://github.com/hukkelas/DSFD-Pytorch-Inference)
  - [Model](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view) (`WIDERFace_DSFD_RES152.pth` - 481Mb) needs to be downloaded from the original repository
    - Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `dsdf/weights`
- [FaceBoxes](https://github.com/sfzhang15/FaceBoxes)
  - [PyTorch implementation](https://github.com/zisianw/FaceBoxes.PyTorch) is used in this project
  - [Model](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI/edit) (`FaceBoxesProd.pth` - 4.1Mb) needs to be downloaded from [PyTorch implementation version](https://github.com/zisianw/FaceBoxes.PyTorch)
    - Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `face_boxes/weights`
- [Haarcascade from OpenCV](https://github.com/opencv/opencv)
  - [Model](./haarcascade/haarcascade_frontalface_default.xml) (1MB) already included in this repository
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
  - Implementation from [this repository](https://github.com/timesler/facenet-pytorch)
- [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
  - [ResNet50](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0) model (112Mb) pretrained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
  - Automatically downloaded with first use (the model is placed @ `$USER$\.insightface\models\`)
- [S3FD](https://github.com/Team-Neighborhood/awesome-face-detection/tree/master/S3FD)
  - [Model](https://drive.google.com/file/d/1Dyr-s3mAQEj-AXCz8YIIYt6Zl3JpjFQ7/view) (`sfd_face.pth` - 90Mb) needs to be downloaded from the repository
    - Trained on  [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `S3FD/weights`
- [YoloFace](https://github.com/sthanhng/yoloface)
  - [Model](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view) (`yolov3-wider_16000.weights` - 246Mb) needs to be downladed from the repository
    - YOLOv3 model trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `yoloface/model-weights`

\
\

### Accuracy

*Reported by the authors*



- Results on validation set of WIDER FACE:

| Model       | Easy Set | Medium Set | Hard Set |
| ----------- | :------: | :--------: | :------: |
| CenterFace  |   93.5   |    92.4    |   87.5   |
| DSFD        |   96.6   |    95.7    |   90.4   |
| FaceBoxes   |    -     |     -      |    -     |
| Haarcascade |    -     |     -      |    -     |
| MTCNN       |    -     |     -      |    -     |
| RetinaFace  |   96.9   |    96.1    |   91.8   |
| S3FD        |   93.7   |    92.4    |   85.2   |
| YoloFace    |    -     |     -      |    -     |





- Results on test set of WIDER FACE:

| Model       | Easy Set | Medium Set | Hard Set |
| ----------- | :------: | :--------: | :------: |
| CenterFace  |   93.2   |    92.1    |   87.3   |
| DSFD        |   96.0   |    95.3    |   90.0   |
| FaceBoxes   |    -     |     -      |    -     |
| Haarcascade |    -     |     -      |    -     |
| MTCNN       |    -     |     -      |    -     |
| RetinaFace  |   96.3   |    95.6    |   91.4   |
| S3FD        |   92.8   |    91.3    |   84.0   |
| YoloFace    |    -     |     -      |    -     |



\

\



## Face recognition

[InsightFace/ArcFace](https://github.com/deepinsight/insightface) recognition model is used to preform face recognition. Faces are saved in a list of recognized faces once they are recognized as a new face. A face is recognized as a new face if none of the other recognized faces doesn't achieve higher similarity than `FACE_CONF_THRESHOLD`. Face recognition can be easily switched on by using `retina_face` detector and setting `retina_face.Recognition = True`.



\

\




## Performance with current settings used to detect faces of volleyball players



| Model       | Seconds per frame |
| ----------- | :---------------: |
| CenterFace  |        1          |
| DSFD        |        48         |
| FaceBoxes   |        0.3        |
| Haarcascade |         1         |
| MTCNN       |         4         |
| RetinaFace  |        90         |
| S3FD        |        20         |
| YoloFace    |        10         |



\

\

## More

[State-of-the-art methods](https://paperswithcode.com/sota/face-detection-on-wider-face-hard) for Face Detection on WIDER Face (Hard) dataset

[Papers with code - Face Detectors](https://paperswithcode.com/task/face-detection)
