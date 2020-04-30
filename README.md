# FaceMask-Detection system using YOLOv3 in Keras

First of all, I must mention that this code used in this project originally is not mine. I was inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3), he wrote this code to train custom YOLOv3 Keras model.

Few modifications are:
1. To load images for detection, I replaced PIL with opencv.
2. Used the opencv font for drawing text on detected images.
3. Wrote seperate detection script for images and videos.

# Dependencies :
- Python 3.6.10
- Keras 2.2.4 `conda install keras=2.2.4`
- Tensorflow 1.15 `conda install tensorflow-gpu=1.15` or `conda install tensorflow`
- OpenCV-python
- Moviepy(for the detection of videos)  `pip install moviepy`
- PIL
- NumPy

Note that keras and tensorflow have their own dependencies. I recommend using [Anaconda](https://www.anaconda.com/) for handlinng the packages.

# Dataset :
I created the dataset for this project. It is divided into two categories ie with_face_mask and without_face_mask. I downloaded the images for both categories from internet and saved them in respective folder in the local disk.

The most important part is the creation of Annotations and bounding boxes; for that purpose I used the [LabelImg](https://github.com/tzutalin/labelImg) annotation tool and created the labels and bounding boxes for each image.

In LabelImg the annotations files are available in two format ie VOC format(XML file) or YOLO format. So, I got the data annotation file in YOLO format directly from labelImg.

