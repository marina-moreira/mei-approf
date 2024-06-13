import tensorflow as tf

from CNN.models.custom_cnn.model_architecture import create_model
from data_preprocessing import preprocess_data

if __name__ == "__main__":
    data_dir = 'dataset/raw/CUB_200_2011/images/'
    train_generator, validation_generator = preprocess_data(data_dir, 'data/processed/images/')

    model = create_model()
    model.fit(train_generator, epochs=20, validation_data=validation_generator)
    model.save('models/custom_cnn/bird_species_model.h5')
