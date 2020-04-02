# Collection of face detection algorithms



- [CenterFace](https://github.com/Star-Clouds/CenterFace) 
  - Model (7.3Mb) already included in this repository
- [DSFD](https://github.com/TencentYoutuResearch/FaceDetection-DSFD) 
  - Implementation from [this repository](https://github.com/hukkelas/DSFD-Pytorch-Inference)
  - [Model](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view) (`WIDERFace_DSFD_RES152.pth` - 481Mb) needs to be downloaded from the original repository
    - Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `dsdf/weights` 
- [FaceBoxes](https://github.com/sfzhang15/FaceBoxes) (`FaceBoxesProd.pth` - 4.1Mb)
  - [PyTorch implementation](https://github.com/zisianw/FaceBoxes.PyTorch) is used in this project
  - [Model](https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI/edit) needs to be downloaded from [PyTorch implementation version](https://github.com/zisianw/FaceBoxes.PyTorch)
    - Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `face_boxes/weights`
- [Haarcascade from OpenCV](https://github.com/opencv/opencv)
  - Model (1MB) already included in this repository
- [MTCNN](https://github.com/ipazc/mtcnn)
  - Implementation from [this repository](https://github.com/timesler/facenet-pytorch)
- [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
  - [ResNet50](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0) model (112Mb) pretrained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) 
  - Automatically downloaded with first use (the model is placed @ `$USER$\.insightface\models\`)
- [S3FD]() 
  - [Model](https://drive.google.com/file/d/1Dyr-s3mAQEj-AXCz8YIIYt6Zl3JpjFQ7/view) (`sfd_face.pth` - 90Mb) needs to be downloaded from the repository 
    - Trained on  [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - Should be placed under `S3FD/weights`
- [YoloFace](https://github.com/sthanhng/yoloface)
  - [Model](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view) (`yolov3-wider_16000.weights` - 246Mb) needs to be downladed from the repository
    - Should be placed under `yoloface/model-weights`
    - YOLOv3 model trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)

