import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.utils import shuffle

samples = []
# with open('data/driving_log.csv') as csvfile:
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)  
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Using all three cameras onboard
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

# Flip the original images and steering angles and keep them as new data
def flip_img(images, angles):
    fliped_images, fliped_angles = [], []
    for image in images:
        fliped_images.append(image)
        fliped_images.append(np.fliplr(image))
    for angle in angles:
        fliped_angles.append(angle)
        fliped_angles.append(angle * (-1))
    return fliped_images, fliped_angles

# Generate data in batches to feed the model
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


#
# Nvidia End-toEnd CNN Architecture
#

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

with tf.device('/GPU:0'):
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, 
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
                        validation_data=validation_generator, 
                        validation_steps=np.ceil(len(validation_samples)/batch_size), 
                        epochs=5, verbose=1)

model.save('model.h5')
print("Model Saved")
exit()