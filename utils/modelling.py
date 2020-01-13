from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

from utils.constants import (
    BATCH_SIZE,
    DROPOUT_RATE,
    NUMBER_OF_EPOCHS,
    NUMBER_OF_UNITS,
    WEIGHTS_FILE,
    SEQUENCE_LENGTH,
)
from utils.common import reshape_and_normalize


def modify_training_and_target_list(training_list, target_list, characters):
    training_ndarray = reshape_and_normalize(
        training_list, (len(training_list), SEQUENCE_LENGTH, 1), len(characters)
    )
    target_ndarray = np_utils.to_categorical(target_list)
    return (training_ndarray, target_ndarray)


def create_two_layer_lstm_model(training_ndarray, target_ndarray):
    model = Sequential()
    model.add(
        LSTM(
            NUMBER_OF_UNITS,
            input_shape=(training_ndarray.shape[1], training_ndarray.shape[2]),
            return_sequences=True,
        )
    )
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NUMBER_OF_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(target_ndarray.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def load_existing_weights_or_train_model(model, training_ndarray, target_ndarray):
    try:
        model.load_weights(WEIGHTS_FILE)
    except:
        model.fit(
            training_ndarray,
            target_ndarray,
            epochs=NUMBER_OF_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        model.save_weights(WEIGHTS_FILE)

    return model


def get_trained_model(training_list, target_list, characters):
    (training_ndarray, target_ndarray) = modify_training_and_target_list(
        training_list, target_list, characters
    )
    model = create_two_layer_lstm_model(training_ndarray, target_ndarray)
    return load_existing_weights_or_train_model(model, training_ndarray, target_ndarray)
