import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

train_dir = '/home/mint/repos/Teeth_Dataset/Training'
validation_dir = '/home/mint/repos/Teeth_Dataset/Testing'
test_dir = '/home/mint/repos/Teeth_Dataset/Validation'

batch_size = 128

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(192, 192),
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(192, 192),
    batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(192, 192),
    batch_size=batch_size)

teeth_cnn = keras.models.Sequential([

        keras.Input(shape=(192, 192, 3)),

        keras.layers.Conv2D(filters = 24, kernel_size = (3, 3), activation = 'relu'),
        keras.layers.AveragePooling2D(pool_size = (2, 2)),

        keras.layers.Conv2D(filters=128, kernel_size = (3, 3), activation = 'relu'),
        keras.layers.AveragePooling2D(pool_size = (2, 2)),

        keras.layers.Conv2D(filters=96, kernel_size = (3, 3), activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (2, 2)), 

        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(units = 96, activation = 'relu'),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(units = 7, activation = 'softmax')
    ])

teeth_cnn.summary()
keras.utils.plot_model(teeth_cnn, rankdir='LR', show_dtype = True)

teeth_cnn.compile(optimizer = 'nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_size = len(train_generator)
val_size = len(validation_generator)
train_steps = int(train_size / batch_size)
val_steps = int(val_size / batch_size)

model_history = teeth_cnn.fit(train_generator, validation_data = validation_generator, epochs = 500, steps_per_epoch = train_steps, validation_steps = val_steps)

loss_acc_df = pd.DataFrame(model_history.history)
loss_acc_df.plot(figsize=(10, 5))

loss, acc = teeth_cnn.evaluate(test_generator)
print(f'loss: {loss}, acc: {acc}')

teeth_cnn.save('teeth_cnn_final.keras')