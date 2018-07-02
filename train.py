from luma.dataset.lsun import LSUNTransform, LSUNGenerator
from luma.models.conv import Encoder, Decoder

import argparse
import cv2
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# Configure logging
validation_logger = logging.getLogger('validation')
validation_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
validation_log_handler = logging.FileHandler('validation.log')
validation_logger.addHandler(console_handler)
validation_logger.addHandler(validation_log_handler)

training_logger = logging.getLogger('training')
training_logger.setLevel(logging.INFO)
train_log_handler = logging.FileHandler('training.log')
training_logger.addHandler(train_log_handler)

tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)


@tfe.implicit_value_and_gradients
def calculate_gradients(image_batch, encoder, decoder):
    batch = tf.constant(image_batch)
    image_encoding = encoder(batch)
    output_image = decoder(image_encoding)
    loss = tf.losses.mean_squared_error(labels=image_batch, predictions=output_image)
    return loss


def evaluate(image_batch, encoder, decoder):
    batch = tf.constant(image_batch)
    image_encoding = encoder(batch)
    output_image = decoder(image_encoding)
    loss = tf.losses.mean_squared_error(labels=image_batch, predictions=output_image)
    return loss


def train_model(train_file, validation_file, validation_interval, width, height, batch_size, n_epochs,
                checkpoint_folder, training_device):
    checkpoint_folder = os.path.join(checkpoint_folder, f'{width}x{height}')
    training_generator = LSUNGenerator(train_file)
    transform = LSUNTransform(image_dimensions=(height, width, 3))
    encoder = Encoder()
    decoder = Decoder()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    checkpointer = tfe.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    best_loss = 1e10
    for epoch in range(n_epochs):
        iteration = 0
        dataset = tf.data.Dataset.from_generator(generator=lambda: training_generator, output_types=tf.string) \
            .map(transform).batch(batch_size)
        for batch in dataset:
            with tf.device(training_device):
                loss, grads_and_vars = calculate_gradients(batch, encoder, decoder)
                optimizer.apply_gradients(grads_and_vars)
            iteration += 1
            training_logger.info(f'Epoch = {epoch}, Iteration = {iteration}, Loss = {loss}')
            if iteration % validation_interval == 0:
                validation_logger.info(f'Epoch: {epoch}, Iteration: {iteration}. Beginning validation pass...')
                validation_generator = LSUNGenerator(validation_file)
                validation_dataset = tf.data.Dataset.from_generator(generator=lambda: validation_generator,
                                                                    output_types=tf.string) \
                    .map(transform).batch(batch_size)
                losses = list()
                for val_batch in validation_dataset:
                    with tf.device(training_device):
                        val_batch = tf.constant(val_batch)
                        loss = evaluate(val_batch, encoder, decoder)
                        losses.append(loss)
                losses = np.array(losses)
                avg_loss = np.mean(losses)
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                std_loss = np.std(losses)
                validation_logger.info(f'avg: {avg_loss}, std: {std_loss}, min: {min_loss}, max: {max_loss}')
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    validation_logger.info(
                        f'Validation loss is best seen so far. Checkpointing to {checkpoint_folder}...')
                    checkpointer.save(checkpoint_folder)


def display(image):
    image = image.numpy()
    image = image * 255
    cv2.imshow('output_image', image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', required=True)
    parser.add_argument('--validation-file', required=True)
    parser.add_argument('--validation-interval', type=int, default=250)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--checkpoint-folder', required=True)
    parser.add_argument('--device', default='/gpu:0')
    args = parser.parse_args()
    train_model(args.training_file, args.validation_file, args.validation_interval, args.width, args.height,
                args.batch_size, args.n_epochs, args.checkpoint_folder, args.device)
