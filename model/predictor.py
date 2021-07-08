import tensorflow as tf
from tensorflow import keras
import model


class NN:
    def make(input_shape, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            NN_model = keras.Sequential([
                keras.layers.Dense(16, activation='relu', input_shape=input_shape),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid',
                                   bias_initializer=output_bias),
            ])

        NN_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=model.metrics)

        return NN_model
