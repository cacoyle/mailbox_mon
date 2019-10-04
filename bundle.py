#!/usr/bin/env python
# coding=utf8

import numpy as np
import os
import pickle

from PIL import Image

# Our use case is simple, keep it simple, stupid.
labels = {
    0: '/opt/mailbox/mailtruck_yes',
    1: '/opt/mailbox/mailtruck_no',
}

data = {
    b'batch_label': 'mailtruck detection images',
    b'labels': [],
    b'data': [],
    b'filenames': []
}

for tag, image_dir in labels.items():
    for image in os.listdir(image_dir):
        path = os.path.join(image_dir, image)
        im = Image.open(path)
        im = (np.array(im))

        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()

        data[b'labels'].append(tag)
        data[b'data'].append(
            # np.array([tag] + list(r) + list(g) + list(b), np.uint8)
            np.array(list(r) + list(g) + list(b), np.uint8)
        )
        data[b'filenames'].append(image)

data[b'data'] = np.asarray(data[b'data'])

pickle.dump(data, open('mailtruck_data_1', 'wb'))
