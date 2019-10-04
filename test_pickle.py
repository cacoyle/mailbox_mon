from matplotlib import pyplot
import numpy as np
import pdb
import pickle
import png
import cv2
from PIL import Image

data_file= 'mailtruck_data_1'
# data_file= 'test_batch'
# data_file= 'data_batch_1' # from cifar10

with open(data_file, 'rb') as data:
    datas = pickle.load(data, encoding='bytes')

# https://realpython.com/storing-images-in-python/
for i, flat_im in enumerate(datas[b'data']):
    pdb.set_trace()
    im_channels = []
    for j in range(3):
        im_channels.append(
            # flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32)) # cifar10 spec
            flat_im[j * 32000: (j + 1) * 32000].reshape((160, 200))
        )

    cv2.imwrite('/opt/mailbox/test_out_cv2.png', np.dstack(im_channels))
    # im = Image.fromarray([im_channels[0]]).save("/opt/mailbox/test_out.jpg")
    # pyplot.imsave(im_channels, 'output.png')
    break
