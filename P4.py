#!/usr/bin/env python
# coding: utf-8

# # Behavioral Cloning

import csv
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
# with open('driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    n = 0
    for line in reader:
        if(n != 0):
            lines.append(line)
        n = n + 1
        print(n)
        
images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    # current_path = 'driving_data/IMG/' + filename
    # image = ndimage.imread(current_path)
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurements)
    
X_train = np.array(images)
y_train = np.array(measurements)



from keras.models import Sequential
from keras.layers import Faltten, Dense

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
exit()

# In[ ]:




