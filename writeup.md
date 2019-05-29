Jupyter Notebook
P4
(autosaved)
Current Kernel Logo
Python [conda env:py3] * 
File
Edit
View
Insert
Cell
Kernel
Help

Behavioral Cloning
# Run only once, to solve the conflict with ROS
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
Data Augmentation
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.utils import shuffle
​
samples = []
# with open('data/driving_log.csv') as csvfile:
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)  
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, 
                                                     test_size=0.2)
​
def multi_cam(batch_sample):
    correction = 0.2
    
    multi_cam_images, multi_cam_angles = [], []
    for i in range(0, 3):
        name = 'data/IMG/' + batch_sample[i].split('/')[-1]
        image = plt.imread(name)
        # if using the left/right/center camera
        if(i == 1): 
            angle = float(batch_sample[3]) + correction
        elif(i == 2): 
            angle = float(batch_sample[3]) - correction
        else: 
            angle = float(batch_sample[3])
        multi_cam_images.append(image)
        multi_cam_angles.append(angle)
    return multi_cam_images, multi_cam_angles
​
​
def flip_img(images, angles):
    fliped_images, fliped_angles = [], []
    for image in images:
        fliped_images.append(image)
        fliped_images.append(np.fliplr(image))
    for angle in angles:
        fliped_angles.append(angle)
        fliped_angles.append(angle * (-1))
    return fliped_images, fliped_angles
​
​
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                
                # Using all three cameras, output 3*1 array of images and angles
                multi_cam_images, multi_cam_angles = multi_cam(batch_sample)
                # flip all the camera images, output 6*1 array of images and angles
                fliped_images, fliped_angles = flip_img(multi_cam_images, multi_cam_angles)
                
                output_images = fliped_images
                output_angles = fliped_angles
                images.extend(output_images)
                angles.extend(output_angles)
                
                #name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                #center_image = plt.imread(name)
                #center_angle = float(batch_sample[3])
                #images.append(center_image)
                #angles.append(center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
​
​
Nvidia End-to-End CNN Architecture
1
import tensorflow as tf
2
from keras.models import Sequential
3
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense
4
​
5
# Set our batch size
6
batch_size = 32
7
​
8
# compile and train the model using the generator function
9
train_generator = generator(train_samples, batch_size=batch_size)
10
validation_generator = generator(validation_samples, batch_size=batch_size)
11
​
12
model = Sequential()
13
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
14
model.add(Cropping2D(cropping=((70,25), (0,0))))
15
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
16
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
17
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
18
model.add(Conv2D(64, (3, 3), activation="relu"))
19
model.add(Conv2D(64, (3, 3), activation="relu"))
20
model.add(Flatten())
21
model.add(Dense(100))
22
model.add(Dense(50))
23
model.add(Dense(10))
24
model.add(Dense(1))
25
​
26
with tf.device('/GPU:0'):
27
    model.compile(loss='mse', optimizer='adam')
28
    model.fit_generator(train_generator, 
29
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
30
                        validation_data=validation_generator, 
31
                        validation_steps=np.ceil(len(validation_samples)/batch_size), 
32
                        epochs=5, verbose=1)
33
​
34
model.save('model.h5')
35
print("Model Saved")
36
# exit()
LeNet Architecture
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
​
# Set our batch size
batch_size = 32
​
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
​
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6, (5, 5), activation="relu", strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation="relu", strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))
​
with tf.device('/GPU:0'):
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, 
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
                        validation_data=validation_generator, 
                        validation_steps=np.ceil(len(validation_samples)/batch_size), 
                        epochs=5, verbose=1)
​
model.save('lenet_model.h5')
print("Model Saved")
# exit()
​
# **Behavioral Cloning** 
​
## Writeup Template
​
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
​
---
​
**Behavioral Cloning Project**
​
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
​
​
[//]: # (Image References)
​
[image1]: ./examples/issue.jpg "Issue with the LeNet Model"
[image2]: ./examples/nvidia-cnn.png "Model Visualization"
[image3]: ./examples/anti-clockwise.jpg "Anti-clockwise Driving"
[image4]: ./examples/clockwise.jpg "Clockwise Driving"
[image5]: ./examples/recover_left.jpg "Recovery from Left"
[image6]: ./examples/recover_right.jpg "Recovery from Right"
[image7]: ./examples/unfliped.jpg "Normal Image"
[image8]: ./examples/fliped1.jpg "Flipped Image"
​
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
​
---
### Files Submitted & Code Quality
​
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
​
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
​
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
​
#### 3. Submission code is usable and readable
​
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
​
### Model Architecture and Training Strategy
​
#### 1. An appropriate model architecture has been employed
​
In the first place, I used a modifed LeNet model to try driving the car autonomously. At most of the time, it can keep the car at the center of the road beautifully. However, at the intersection of the main road and the branch, the network tends to drive the car into that unconstructed part of the road (on the right):
​
![alt text][image1]
​
Then, I tried to record driving on the track both anti-clockwise and clockwise, hoping that it can generalize the model. The result showed that it really worked, through the car will pop onto the ledges on the right hand side of the road during that turn. 
​
To further improve the model, I decided to adopt a new architecture, which is the one nvidia published in their [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and achieved very impressive results. 
​
The model includes RELU layers to introduce nonlinearity (code line 97-101), and the data is normalized in the model using a Keras lambda layer (code line 95), and corpped to a shape of 65x320x3 (code line 96).
​
#### 2. Final Model Architecture
​
Here is a visualization of the final model architecture (model.py lines 94-106)
​
![alt text][image2]
​
#### 3. Model parameter tuning
​
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).
​
#### 4. Appropriate training data
​
To capture good driving behavior, I first recorded two laps on track one, driving anti-clockwise. Here is an example image of center lane driving:
​
![alt text][image3]
​
I then recorded another two laps on the same track, clockwise: 
​
![alt text][image4]
​
During each lap, I tried to maintain the vehicle on the center of the road. Sometimes(Due to my bad driving skill), the vehicle is a bit off, and I use these moments to train the vehicle to recover from the left side and right sides of the road back to center. These images show what a recovery looks:
​
![alt text][image5]
![alt text][image6]
​
​
To augment the data sat, I also flipped images and angles thinking that this would prevent the predictions biased towards one direction. For example, here is an image that has then been flipped:
​
![alt text][image7]
![alt text][image8]
​
After the collection process, I had 27600 number of sets of data(including images from the center, left and right cameras, and their fliped copy). I randomly shuffled the data set and put 20% of the data into a validation set. 
​
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the changes in loss after each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
​
