import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.__decoder_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=512,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=256,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=128,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=64,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=32,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=3,
                                            padding='SAME',
                                            strides=2,
                                            kernel_size=3,
                                            activation=tf.nn.tanh)
        ])

    def call(self, inputs):
        return self.__decoder_layers(inputs)
