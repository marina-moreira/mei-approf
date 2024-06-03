import os
import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import array_to_img


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


def preprocess_and_save_images(data_dir, target_dir):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subset in ['training', 'validation']:
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=1,
            class_mode='categorical',
            subset=subset,
            shuffle=False
        )

        subset_dir = os.path.join(target_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)

        for i in range(len(generator)):
            x, y = generator.next()
            img = array_to_img(x[0])
            label = generator.filenames[i].split('/')[0]
            img_save_dir = os.path.join(subset_dir, label)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            img.save(os.path.join(img_save_dir, generator.filenames[i].split('/')[-1]))



if __name__ == "__main__":
    data_dir = '../dataset/raw/CUB_200_2011/images/'
    target_dir = '../dataset/processed/images/'
    preprocess_data(data_dir, target_dir)
