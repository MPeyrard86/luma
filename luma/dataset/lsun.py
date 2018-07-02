import tensorflow as tf


class LSUNGenerator:
    def __init__(self, source_file):
        self._source_file = source_file

    def __iter__(self):
        with open(self._source_file, 'r') as src:
            for image_path in src:
                yield tf.constant(image_path.strip())


class LSUNTransform:
    def __init__(self, image_dimensions=(256, 256, 3)):
        self._height, self._width, self._channels = image_dimensions

    def __call__(self, image_path):
        jpg_file = tf.read_file(image_path)
        image = tf.image.decode_image(jpg_file)
        image = tf.image.resize_image_with_crop_or_pad(image, self._height, self._width)
        image = tf.cast(image, tf.float32)
        image = image/255.0
        return image
