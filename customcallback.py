#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
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
        out = np.array(self.model.predict(inp))

        shape = out.shape

        img = out.reshape(-1)
        idx = np.where(np.logical_and(img > 0.2, img < 0.8))[0]
        if len(idx) > 0:
            img[idx] = -1
        img = img.reshape(shape)
        img[img != -1] = 0
        img[img == -1] = 255
        img = np.floor(img)
        return img

    def on_epoch_end(self, epoch, logs={}):

        y_test = load_img(self._y_test)
        y_test = y_test.resize(self.input_size)
        y_test = img_to_array(y_test)

        img = load_img(self._x_test)
        img = img.resize(self.input_size)
        img_org_out = img_to_array(img)
        img = np.array([img_org_out])

        out = self.custom_pred(img)

        img_out = out[0]

        mask = np.zeros_like(img_org_out)
        mask[:, :, :] = img_out
        mask[mask > 0] = 255
        img_org_out[mask == 0] = 0

        fig2, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 8))

        a0.imshow(img_org_out / 256.0, label="original")

        a1.imshow(mask / 256.0, label="pred")

        a2.imshow(y_test / 256.0, label="real")

        outfile = os.path.join(
            self._savedir, f"{self._model_name}_{epoch}.png")

        plt.savefig(outfile)
