import os

from keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Data Preprocessing Script:
Contains functions to preprocess the image data, which are likely used in the notebooks.
Includes functions for data augmentation and creating data generators for training and validation.
'''


def create_data_generators(data_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Classes found: {train_generator.class_indices}")
    print(f"Number of classes in training generator: {train_generator.num_classes}")
    print(f"Number of classes in validation generator: {validation_generator.num_classes}")

    return train_generator, validation_generator


def preprocess_and_save_images(data_dir, target_dir):
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalizes the pixel values of the images.
        rotation_range=40,  # Randomly rotates images. Images can be rotated by up to 40 degrees.
        width_shift_range=0.2,  # Randomly shifts images horizontally.Images can be shifted up to 20% of the width
        height_shift_range=0.2,  # Randomly shifts images vertically.Images can be shifted up to 20% of the height.
        shear_range=0.2,  # Applies shearing transformations. hear transformations alter the image in a way that shifts one part of the image in a direction parallel to the opposing axis, which can help in making the model robust to such transformations.
        zoom_range=0.2,  # Randomly zooms in on images.Images can be zoomed in by up to 20%.
        horizontal_flip=True,  # Randomly flips images horizontally.This can double the dataset by flipping images.
        fill_mode='nearest',  # Fills in new pixels after transformations.When transformations (like rotation or shifting) create empty areas in the image, they are filled in using the nearest pixel values.
        validation_split=0.2  # Splits the data into training and validation sets. 20% of the data is reserved for validation, which helps in evaluating the model's performance on unseen data.
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
            x, y = next(generator)
            img = array_to_img(x[0])
            label = generator.filenames[i].split(os.sep)[0]
            img_save_dir = os.path.join(subset_dir, label)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            img.save(os.path.join(img_save_dir, os.path.basename(generator.filenames[i])))