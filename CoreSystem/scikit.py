#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage import data
import os

matplotlib.rcParams['font.size'] = 18




filename = '/home/josuehfa/Downloads/2020-03-24--09 47 20.png'


images = ['astronaut']
for name in images:
    caller = getattr(data, name)
    image = caller()
    plt.figure()
    plt.title(name)
    if image.ndim == 2:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)

plt.show()