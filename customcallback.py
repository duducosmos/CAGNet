#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import Callback


class PlotTestImages(Callback):

    def __init__(self, x_test, y_test, input_size, model_name, savedir):
        self._x_test = x_test
        self._y_test = y_test

        self._model_name = model_name
        self._savedir = savedir
        self.input_size = input_size

    def custom_pred(self, inp):
        predicted_map = self.model.predict(inp)
        img = (predicted_map*255).astype('uint8')
        return img

    def image_preprocess(self, image):
        image = image / 255
        image = image.astype(np.float32)
        return image

    def read_image(self, img_dir, mode='bgr'):
        image = cv2.imread(img_dir)
        if mode == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = self.image_preprocess(image)
        name = os.path.basename(img_dir)
        return image, name

    def write_image(self, predicted_map, image, name, epoch):
        name = os.path.splitext(name)[0] + f'_{epoch}.png'
        predicted_map = (predicted_map*255).astype('uint8')
        plotimages = os.path.join(self._savedir, "plottestimages")
        os.makedirs(plotimages, exist_ok=True)

        image[predicted_map < 255] = 0

        original, original_name = self.read_image(self._y_test)

        fig2, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 8))

        a0.imshow(image, label="original")

        a1.imshow(predicted_map, label="pred")

        a2.imshow(original, label="real")

        outfile = os.path.join(plotimages, name)

        plt.savefig(outfile)
        plt.close()

        # cv2.imwrite(os.path.join(plotimages, name) , predicted_map)

    def on_epoch_end(self, epoch, logs={}):

        image, name = self.read_image(self._x_test)
        pred = self.model.predict(np.expand_dims(image, 0))

        mask = pred[0, ..., 1]

        self.write_image(mask, image, name, epoch)
