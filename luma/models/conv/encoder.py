import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self._encoder_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        ])

    def call(self, inputs):
        return self._encoder_layers(inputs)
