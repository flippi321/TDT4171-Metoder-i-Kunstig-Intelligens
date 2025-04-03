import pickle
from typing import Dict, List, Any, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, BatchNormalization, Dropout, LeakyReLU

# Ensure TensorFlow uses GPU if available
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available and will be used.")
else:
    print("No GPU found. Running on CPU.")

def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)
    return data

def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data

def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward", lr=1e-3, epochs = 1) -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """
    vocab_size = data["vocab_size"]
    embedding_dim = 128
    maxlen = data["x_train"].shape[1]

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))

    if model_type == "feedforward":
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer="he_normal"))
        model.add(LeakyReLU(negative_slope=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(128, kernel_initializer="he_normal"))
        model.add(LeakyReLU(negative_slope=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(64, activation=tf.keras.activations.swish, kernel_initializer="he_normal"))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
    elif model_type == "recurrent":
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(64, activation='relu'))
    else:
        raise ValueError("Invalid model_type. Choose 'feedforward' or 'recurrent'.")

    model.add(Dense(1, activation='sigmoid'))
    optim = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(data["x_train"], data["y_train"], epochs=epochs, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    return accuracy

def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()
