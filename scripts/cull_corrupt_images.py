import argparse
import cv2


def cull_corrupted_images(input_filename, output_filename):
    valid_images = list()
    corrupted_count = 0
    with open(input_filename, 'r') as input_file:
        for image_path in input_file:
            image_path = image_path.strip()
            im = cv2.imread(image_path, 0)
            if im is not None:
                valid_images.append(image_path)
            else:
                corrupted_count += 1
    with open(output_filename, 'w') as output_file:
        for image_path in valid_images:
            output_file.write(image_path)
    print(f'Culled {corrupted_count} images!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    cull_corrupted_images(args.input, args.output)
