# WildFire-Predictor
This project is at the forefront of this research effort, seeking to develop novel approaches to predict the occurrence and behavior of wildfires. Using advanced machine learning algorithms, we will analyze a vast amount of satellite data imagery.


## Installation

Open up https://www.kaggle.com/

Click on 'Register' if you don't have a kaggle account

Else 'Sign In' using your account

Once logged in copy & paste this link "https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset" into search bar

You should now see the "Wildfire Prediction Dataset (Satellite Images)" page by ABDELGHANI AABA

Click on "New Notebook" which should take you to brand new notebook that has the dataset of wildfire satellite images already installed in "Input" on the right side below "Data"

Now copy and paste the code given to you by me and it should run properly.
## Code

``` 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

``` 


#### 1. Importing Libraries

``` 
import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import pandas as pd

``` 

``` 
for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))

``` 
**OUTPUT**:

![WP1](https://github.com/krish1348/WildFire-Predictor/assets/90926847/85e3cf14-0d09-49a4-b9b3-45294b96f2c7)




#### 2. Gathering Data

``` 
train_path = "../input/wildfire-prediction-dataset/train"
valid_path = "../input/wildfire-prediction-dataset/valid"
test_path = "../input/wildfire-prediction-dataset/test"

``` 
``` 
image_shape = (350,350,3)
N_CLASSES = 2
BATCH_SIZE = 256

# loading training data and rescaling it using ImageDataGenerator
train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')

# loading validation data and rescaling it using ImageDataGenerator
valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')

# loading test data and rescaling it using ImageDataGenerator
test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')
```

**OUTPUT**:

![WP2](https://github.com/krish1348/WildFire-Predictor/assets/90926847/53699596-0275-4778-bb86-5f5e063aaa2d)




#### 3. Building the model

``` 
weight_decay = 1e-3
first_model = Sequential([
    Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 32 , kernel_size = 2, activation = 'relu',
           kernel_regularizer = regularizers.l2(weight_decay)),
    MaxPooling2D(pool_size = 2),
    
    Dropout(0.4),
    Flatten(),
    Dense(300,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='softmax')
])

first_model.summary()

``` 
**OUTPUT**:


