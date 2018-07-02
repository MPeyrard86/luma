import argparse
import cv2
import os
import random


def check_image_validity(image_path):
    im = cv2.imread(image_path, 0)
    return im is not None


def prepare_dataset(dataset_folder, output_filename):
    image_paths = list()
    processed_count = 0
    corrupt_count = 0
    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for f in filenames:
            if f.endswith('.webp'):
                image_path = os.path.join(dirpath, f)
                if check_image_validity(image_path):
                    image_paths.append(image_path)
                else:
                    corrupt_count += 1
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f'Procssed {processed_count} images. {corrupt_count} were corrupt.')
    print('Shuffling...')
    random.shuffle(image_paths)
    print('Writing out...')
    with open(output_filename, 'w') as output_file:
        for p in image_paths:
            output_file.write(p + os.linesep)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()
    prepare_dataset(args.dataset_folder, args.output_file)
