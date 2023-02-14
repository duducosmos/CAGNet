import os
import cv2
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from model import cagnet_model
tf.keras.backend.set_image_data_format('channels_last')


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_model", type=str, default='VGG16', choices=['VGG16', 'ResNet50', 'NASNetMobile',
                                                                                'NASNetLarge'])
    parser.add_argument("--local_weights", type=str,
                        required=True,  default=None)
    parser.add_argument("--backbone_weights", type=str,
                        default='imagenet', choices=['imagenet', 'scratch'])
    parser.add_argument("--input_shape", type=tuple, default=(480, 480, 3))
    parser.add_argument("--input_dir", type=str,
                        required=True, default='./data/')
    parser.add_argument("--save_dir", type=str, required=True, default='./save/',
                        help='The path to save the predicted saliency maps')
    return parser.parse_args()


def image_preprocess(image):
    image = image / 255
    image = image.astype(np.float32)
    return image


def read_image(img_dir, dsize=(480, 480), mode='bgr'):
    image = cv2.imread(img_dir)
    if mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_o = cv2.resize(image, dsize)
    image = image_preprocess(image_o)
    name = os.path.basename(img_dir)
    return image_o, image, name


def write_image(predicted_map, image_o, name, save_dir='save'):

    predicted_map = (predicted_map*255).astype('uint8')

    image_o[predicted_map == 0] = 0
    rgba = cv2.cvtColor(image_o, cv2.COLOR_RGB2RGBA)
    rgba[:, :,  3] = predicted_map

    name = os.path.splitext(name)[0] + '.png'
    cv2.imwrite(os.path.join(save_dir, name), rgba)


if __name__ == "__main__":

    cfg = args()
    model = cagnet_model(cfg.backbone_model, cfg.input_shape,
                         backbone_weights=cfg.backbone_weights)

    if cfg.local_weights is not None and os.path.exists(cfg.local_weights):
        model.load_weights(cfg.local_weights)
        print(f"Model loaded {cfg.local_weights}")

    input_height, input_width = model.input.shape[1:3]

    all_images = glob(os.path.join(cfg.input_dir, '*'))
    print('found {} images'.format(len(all_images)))
    for img_dir in all_images:
        image_o, image, name = read_image(img_dir, (input_width, input_height))
        predicted_map = model.predict(np.expand_dims(image, 0))[0, ..., 1]
        write_image(predicted_map, image_o, name, cfg.save_dir)
