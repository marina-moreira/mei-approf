import tensorflow as tf
from data_preprocessing import preprocess_data


def evaluate_model(model_path, data_dir):
    model = tf.keras.models.load_model(model_path)
    _, validation_generator = preprocess_data(data_dir, 'data/processed/images/')
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    model_path = 'models/custom_cnn/bird_species_model.h5'
    data_dir = 'dataset/raw/CUB_200_2011/images/'
    evaluate_model(model_path, data_dir)
