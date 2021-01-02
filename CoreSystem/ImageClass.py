import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2hsv
from skimage import io

#original = data.astronaut()
original = io.imread('/home/josuehfa/System/CoreSystem/TestMap2.png')[:,:,:3]


grayscale = rgb2gray(original)

image_resized = resize(original, (150,150),anti_aliasing=True)

fig1, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
ax[2].imshow(image_resized, cmap=plt.cm.gray)
ax[2].set_title("Resized")

fig1.tight_layout()


#original = data.coffee()
image_resized = resize(original, (150,150),anti_aliasing=True)
hsv_img = rgb2hsv(original)
hue_img = hsv_img[:, :, 0]
sat_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

fig2, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))

ax0.imshow(image_resized)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(sat_img)
ax2.set_title("Sat channel")
ax2.axis('off')
ax3.imshow(value_img)
ax3.set_title("Value channel")
ax3.axis('off')

fig2.tight_layout()


hsv_img = rgb2hsv(image_resized)
hue_img = hsv_img[:, :, 0]
sat_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

fig3, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))

ax0.imshow(image_resized)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(sat_img)
ax2.set_title("Sat channel")
ax2.axis('off')
ax3.imshow(value_img)
ax3.set_title("Value channel")
ax3.axis('off')

fig3.tight_layout()
plt.show()