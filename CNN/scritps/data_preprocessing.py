import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_data(data_dir, target_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    return train_generator, validation_generator


if __name__ == "__main__":
    data_dir = 'dataset/raw/CUB_200_2011/images/'
    target_dir = 'dataset/processed/images/'
    preprocess_data(data_dir, target_dir)
