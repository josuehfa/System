import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2hsv
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage import data
from skimage.feature import match_template

#original = data.astronaut()
pampShape = io.imread('/home/josuehfa/System/CoreSystem/PampulhaShape.png')[:,:,:3]
popDensity = io.imread('/home/josuehfa/System/CoreSystem/PopulationDensity.png')[:,:,:3]
bhStreets = io.imread('/home/josuehfa/System/CoreSystem/BeloHorizonteStreets_Edited2.png')[:,:,:3]



shape_resized = rgb2gray(resize(pampShape, (pampShape.shape[0]//2,pampShape.shape[1]//2),anti_aliasing=True))
density_resized = rgb2gray(resize(popDensity, (popDensity.shape[0]//2,popDensity.shape[1]//2),anti_aliasing=True))

result_density = match_template(density_resized, shape_resized)
ij = np.unravel_index(np.argmax(result_density), result_density.shape)
xdensity, ydensity = ij[::-1]

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3, sharex=ax2, sharey=ax2)

ax1.imshow(shape_resized, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(density_resized, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hdensity, wdensity = shape_resized.shape
rect = plt.Rectangle((xdensity, ydensity), wdensity, hdensity, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result_density)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(xdensity, ydensity, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


shape_resized = rgb2gray(resize(pampShape, (pampShape.shape[0]+pampShape.shape[0]//4,pampShape.shape[1]+pampShape.shape[0]//4),anti_aliasing=True))
street_resized = rgb2gray(resize(bhStreets, (bhStreets.shape[0]*2,bhStreets.shape[1]*2),anti_aliasing=True))

result_street = match_template(street_resized, shape_resized)
ij = np.unravel_index(np.argmax(result_street), result_street.shape)
xstreet, ystreet = ij[::-1]


ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6, sharex=ax4, sharey=ax5)

ax4.imshow(shape_resized, cmap=plt.cm.gray)
ax4.set_axis_off()
ax4.set_title('template')

ax5.imshow(street_resized, cmap=plt.cm.gray)
ax5.set_axis_off()
ax5.set_title('image')
# highlight matched region
hstreet, wstreet = shape_resized.shape
rect = plt.Rectangle((xstreet, ystreet), wstreet,hstreet, edgecolor='r', facecolor='none')
ax5.add_patch(rect)

ax6.imshow(result_street)
ax6.set_axis_off()
ax6.set_title('`match_template`\nresult')
# highlight matched region
ax6.autoscale(False)
ax6.plot(xstreet, ystreet, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


plt.show()


#Proporção entre as imagens
h_prop = hdensity*2/hstreet + hdensity*2/hstreet*0.052
w_prop = wdensity*2/wstreet + wdensity*2/wstreet*0.028

#Modificando o ponto (x,y) para o novo tamanho da imagem
xstreet = round(xstreet*w_prop)
ystreet = round(ystreet*h_prop)

#Modificando o tamanho da imagem para da match com a segunda imagem
street_prop = resize(street_resized, (round(street_resized.shape[0]*h_prop),round(street_resized.shape[1]*w_prop)),anti_aliasing=True)

#Invertendo o valor do street, nesse caso as ruas vão valer 0 e anularão esses pontos na segunda imagem
street_prop = util.invert(street_prop)
#Criando uma mascara: Se for maior que 0 True, else False ( Nesse caso, as ruas são False e o resto True)
streetMask = street_prop > 0.75

popDensity_gray = rgb2gray(popDensity)
#Criando uma mascara para a densidade populacional, se for maior que 0 True, else False (Nesse caso o contorno é False e o interior True)
popMask = popDensity_gray > 0
io.imshow(popMask)
plt.show()

#Como quanto maior a populacao mais escura a região, inverte-se e multiplicasse pela mascara para que 
#Os contornos passem a ter o menor valor (em geral são divisas de bairros)
popDensity_gray = (util.invert(popDensity_gray))*popMask
io.imshow(popDensity_gray)
plt.show()

x_cont = 0
for idx in range((xdensity*2 - xstreet),((xdensity*2 - xstreet) + street_prop.shape[1])):
    y_cont = 0
    for idy in range((ydensity*2 - ystreet),((ydensity*2 - ystreet) + street_prop.shape[0])):
        popDensity[idy][idx] = streetMask[y_cont][x_cont]*popDensity_gray[idy][idx]*popDensity[idy][idx]*10
        popDensity_gray[idy][idx] = streetMask[y_cont][x_cont]*popDensity_gray[idy][idx]
        y_cont = y_cont + 1
    x_cont = x_cont + 1

popDensity = util.invert(popDensity)
io.imshow(popDensity)
plt.show()

popDensity_gray  = util.invert(popDensity_gray)*100
io.imshow(popDensity_gray)
plt.show()



#grayscale = rgb2gray(original)

#image_resized = resize(original, (150,150),anti_aliasing=True)

#fig1, axes = plt.subplots(1, 3, figsize=(8, 4))
#ax = axes.ravel()

#ax[0].imshow(original)
#ax[0].set_title("Original")
#ax[1].imshow(grayscale, cmap=plt.cm.gray)
#ax[1].set_title("Grayscale")
#ax[2].imshow(image_resized, cmap=plt.cm.gray)
#ax[2].set_title("Resized")

#fig1.tight_layout()


#original = data.coffee()
#image_resized = resize(original, (150,150),anti_aliasing=True)
#hsv_img = rgb2hsv(original)
#hue_img = hsv_img[:, :, 0]
#sat_img = hsv_img[:, :, 1]
#value_img = hsv_img[:, :, 2]

#fig2, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))

#ax0.imshow(image_resized)
#ax0.set_title("RGB image")
#ax0.axis('off')
##ax1.imshow(hue_img, cmap='hsv')
#ax1.set_title("Hue channel")
#ax1.axis('off')
#ax2.imshow(sat_img)
#ax2.set_title("Sat channel")
#ax2.axis('off')
#ax3.imshow(value_img)
#ax3.set_title("Value channel")
#ax3.axis('off')#

#fig2.tight_layout()


##hsv_img = rgb2hsv(image_resized)
##hue_img = hsv_img[:, :, 0]
#sat_img = hsv_img[:, :, 1]
#value_img = hsv_img[:, :, 2]

#fig3, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))

##ax0.imshow(image_resized)
# ax0.set_title("RGB image")
# ax0.axis('off')
# ax1.imshow(hue_img, cmap='hsv')
# ax1.set_title("Hue channel")
# ax1.axis('off')
# ax2.imshow(sat_img)
# ax2.set_title("Sat channel")
# ax2.axis('off')
# ax3.imshow(value_img)
# ax3.set_title("Value channel")
# ax3.axis('off')

# fig3.tight_layout()
# plt.show()