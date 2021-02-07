import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray,gray2rgb
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
pampShape = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/PampulhaShape.png')[:,:,:3]
#popDensity = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/PopulationDensity.png')[:,:,:3]
popDensity = io.imread('/home/josuehfa/Pictures/popDensityNew.png')[:,:,:3]
bhStreets = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/BeloHorizonteStreets_Edited2.png')[:,:,:3]



shape_resized = rgb2gray(resize(pampShape, (pampShape.shape[0],pampShape.shape[1]),anti_aliasing=True))
density_resized = rgb2gray(resize(popDensity, (round(popDensity.shape[0]*4.4),round(popDensity.shape[1]*4.4)),anti_aliasing=True))

result_density = match_template(density_resized, shape_resized)
ij = np.unravel_index(np.argmax(result_density), result_density.shape)
xdensity, ydensity = ij[::-1]

fig1 = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(shape_resized, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('Template')


ax2.imshow(density_resized, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('Population Density')
# # highlight matched region
hdensity, wdensity = shape_resized.shape
rect = plt.Rectangle((xdensity, ydensity), wdensity, hdensity, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result_density)
ax3.set_axis_off()
ax3.set_title('Match Result')
# highlight matched region
ax3.autoscale(False)
ax3.plot(xdensity, ydensity, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


shape_resized = rgb2gray(resize(pampShape, (pampShape.shape[0]+pampShape.shape[0]//4,pampShape.shape[1]+pampShape.shape[0]//4),anti_aliasing=True))
street_resized = rgb2gray(resize(bhStreets, (bhStreets.shape[0]*2,bhStreets.shape[1]*2),anti_aliasing=True))

result_street = match_template(street_resized, shape_resized)
ij = np.unravel_index(np.argmax(result_street), result_street.shape)
xstreet, ystreet = ij[::-1]

fig2 = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax5 = plt.subplot(1, 3, 2)
ax6 = plt.subplot(1, 3, 3)

ax1.imshow(shape_resized, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('Template')

ax5.imshow(street_resized, cmap=plt.cm.gray)
ax5.set_axis_off()
ax5.set_title('Streets of BH')
# # highlight matched region
hstreet, wstreet = shape_resized.shape
rect = plt.Rectangle((xstreet, ystreet), wstreet,hstreet, edgecolor='r', facecolor='none')
ax5.add_patch(rect)

ax6.imshow(result_street)
ax6.set_axis_off()
ax6.set_title('Match Result')
# highlight matched region
ax6.autoscale(False)
ax6.plot(xstreet, ystreet, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)




#Proporção entre as imagens (+ hard ajust)
h_prop = hdensity/hstreet + hdensity/hstreet*0.078
w_prop = wdensity/wstreet + wdensity/wstreet*0.01


#Proporção entre as imagens
#h_prop = hdensity*2/hstreet + hdensity*2/hstreet*0.052
#w_prop = wdensity*2/wstreet + wdensity*2/wstreet*0.028

#h_prop = hdensity*2/hstreet + hdensity*2/hstreet*0.052
#w_prop = wdensity*2/wstreet + wdensity*2/wstreet*0.028

#Modificando o ponto (x,y) para o novo tamanho da imagem
xstreet = round(xstreet*w_prop)
ystreet = round(ystreet*h_prop)

#Modificando o tamanho da imagem para da match com a segunda imagem
street_prop = resize(street_resized, (round(street_resized.shape[0]*h_prop),round(street_resized.shape[1]*w_prop)),anti_aliasing=True)

#Invertendo o valor do street, nesse caso as ruas vão valer 0 e anularão esses pontos na segunda imagem
street_prop = util.invert(street_prop)
#Criando uma mascara: Se for maior que 0 True, else False ( Nesse caso, as ruas são False e o resto True)
streetMask = street_prop > 0.4
io.imshow(streetMask)
plt.show()


popDensity_gray = rgb2gray(density_resized)
#Criando uma mascara para a densidade populacional, se for maior que 0 True, else False (Nesse caso o contorno é False e o interior True)
popMask = density_resized > 0.05
#io.imshow(popMask)
#plt.show()

#Niveis:
#0.901 - Menor densidade
#0.773
#0.664
#0.556
#0.465 - Maior densidade

#Como quanto maior a populacao mais escura a região, inverte-se e multiplicasse pela mascara para que 
#Os contornos passem a ter o menor valor (em geral são divisas de bairros)
popDensity_gray = (util.invert(popDensity_gray))*popMask
#io.imshow(popDensity_gray)
#plt.show()

x_cont = 0
for idx in range((xdensity - xstreet),((xdensity - xstreet) + street_prop.shape[1])):
    y_cont = 0
    for idy in range((ydensity - ystreet),((ydensity - ystreet) + street_prop.shape[0])):
        #popDensity[idy][idx] = streetMask[y_cont][x_cont]*popDensity_gray[idy][idx]*popDensity[idy][idx]*10
        #popDensity_gray[idy][idx] = streetMask[y_cont][x_cont]*popDensity_gray[idy][idx]
        street_prop[y_cont][x_cont] = streetMask[y_cont][x_cont]*popDensity_gray[idy][idx]
        y_cont = y_cont + 1
    x_cont = x_cont + 1

#street_prop = util.invert(street_prop)
io.imshow(street_prop)
plt.show()

#popDensity_gray  = util.invert(popDensity_gray)
#io.imshow(popDensity_gray)
#plt.show()


#street_prop = np.multiply(street_prop, np.where(street_prop >= 0.1, 110, 1))
#io.imsave('popCalculatedFinal.png',street_prop) ###



#street_prop = resize(street_prop, (200,200),anti_aliasing=True)


#street_prop*=((street_prop >= 0.3)*1000 + (street_prop < 0.3)*100)

lat_lon_i = (-19.829752116279057, -44.02262249999998)
lat_lon_s = (-19.943540209302327, -43.90054215000001)

lat_rsoares = -19.922752
lon_rsoares = -43.945150
lat_pirulito =-19.919133
lon_pirulito = -43.938626


x_rsoares_street = 933
y_rsoares_street = 1137
x_pirulito_street = 1013
y_pirulito_street = 1091
shape_street = (1389, 1471)



x_r_percent_street = x_rsoares_street/shape_street[1]
x_p_percent_street = x_pirulito_street/shape_street[1]
y_r_percent_street = y_rsoares_street/shape_street[0]
y_p_percent_street = y_pirulito_street/shape_street[0]


x_r_percent_street = x_rsoares_street/shape_street[1]
x_p_percent_street = x_pirulito_street/shape_street[1]
y_r_percent_street = y_rsoares_street/shape_street[0]
y_p_percent_street = y_pirulito_street/shape_street[0]


lat_diff = lat_rsoares - lat_pirulito
lon_diff = lon_rsoares - lon_pirulito
x_diff = (x_r_percent_street - x_p_percent_street)*street_prop.shape[1]
y_diff = (y_r_percent_street - y_p_percent_street)*street_prop.shape[0]

lat_to_y =  lat_diff/y_diff
lon_to_x = lon_diff/x_diff


pnt_i = (0,0)
pnt_s = (1389, 1471)

pnt_i_y = pnt_i[0] - y_r_percent_street*street_prop.shape[0]
pnt_i_x = pnt_i[1] - x_r_percent_street*street_prop.shape[1]

pnt_i_lat = pnt_i_y*lat_to_y + lat_rsoares
pnt_i_lon = pnt_i_x*lon_to_x + lon_rsoares

pnt_s_y = pnt_s[0] - y_r_percent_street*street_prop.shape[0]
pnt_s_x = pnt_s[1] - x_r_percent_street*street_prop.shape[1]

pnt_s_lat = pnt_s_y*lat_to_y + lat_rsoares
pnt_s_lon = pnt_s_x*lon_to_x + lon_rsoares

print(pnt_i_lon, pnt_i_lat)
print(pnt_s_lon, pnt_s_lat)


test = (-19.869245, -43.963622)


x_test = test[1] - lon_rsoares
y_test = test[0] - lat_rsoares

y_value = y_test/lat_to_y + y_r_percent_street*street_prop.shape[0]
x_value = x_test/lon_to_x + x_r_percent_street*street_prop.shape[1]



fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.imshow(street_prop)
ax1.set_axis_off()
ax1.autoscale(False)
ax1.plot(x_value, y_value, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


x = np.linspace(pnt_i_lon,pnt_s_lon,street_prop.shape[1])
y = np.linspace(pnt_i_lat,pnt_s_lat,street_prop.shape[0])

#ax2.imshow(street_prop)
ax2.set_axis_off()
ax2.autoscale(False)
ax2.contourf(x, y, street_prop)
ax2.plot(test[1], test[0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
ax2.plot(pnt_i_lon, pnt_i_lat, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
ax2.plot(pnt_s_lon, pnt_s_lat, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()


import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45,0.45],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map","2D CostMap"),
            specs=[[{"type": "Scattermapbox"},{"type": "contour"}]])
#0
#Trace for solution  
fig.append_trace(go.Scattermapbox(
    lat=[],
    lon=[],
    name='Solution Path',
    mode='markers+lines',
    marker=dict(size=5, color='blue')
    ),row=1,col=1)  


region = [(-19.829752116279057, -44.02262249999998),
          (-19.829752116279057, -43.90054215000001),
          (-19.943540209302327, -43.90054215000001),
          (-19.943540209302327, -44.02262249999998)]

lat_aux = [reg[0] for reg in region]
lon_aux = [reg[1] for reg in region]
fig.append_trace(go.Scattermapbox(
    lon = lon_aux, 
    lat = lat_aux,
    fill = "toself",
    name='Area of Flight',
    marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }),row=1,col=1)

#5
#Contour plot
x = np.linspace(pnt_i_lon,pnt_s_lon,shape_street[1])
y = np.linspace(pnt_i_lat,pnt_s_lat,shape_street[0])
fig.append_trace(go.Contour(
    z=street_prop,
    x=x, 
    y=y,
    name='Cost Time: 0',
    line_smoothing=0,
    colorscale='dense',
    contours=dict(
        start=street_prop.min(), 
        end=street_prop.max(), 
        size=1, 
        showlines=False)
    ), row=1, col=2)
#Fixa o eixo dos plots
fig.update_layout(
    mapbox = {
        #'style': "outdoors",
        'style': "stamen-terrain",
        'center': {'lon': -43.9520, 'lat': -19.8997 },
        #'accesstoken': token,
        'zoom': 11},
    title_text="UAS Path Generation", hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="top",
        #y=1.02,
        xanchor="right",
        x=1
    ),
    updatemenus=[dict( type="buttons",
        showactive= False,
        buttons=[
            dict(label="Play",
                method="animate",
                args=[[None], 
                    dict(frame= { "duration": 50},
                        fromcurrent= True,
                        mode='immediate', 
                        transition= {"duration":10, "easing": "linear"}
                        )]),
            dict(label='Pause',
                method='animate',
                args= [ [None],
                    dict(frame= { "duration": 0},
                        fromcurrent= True,
                        mode='immediate', 
                        transition= {"duration": 0, "easing": "linear"}
                                )]
            )]
        )])

plotly.offline.plot(fig, filename='test.html')


x_rsoares = 1497
y_rsoares = 1840
x_pirulito = 1577
y_pirulito = 1796
shape = (3571, 2621)

x_r_percent = x_rsoares/shape[1]
x_p_percent = x_pirulito/shape[1]
y_r_percent = y_rsoares/shape[0]
y_p_percent = y_pirulito/shape[0]


x_r_percent = x_rsoares/shape[1]
x_p_percent = x_pirulito/shape[1]
y_r_percent = y_rsoares/shape[0]
y_p_percent = y_pirulito/shape[0]

popDensity = resize(popDensity_gray, (700,700),anti_aliasing=True)

lat_diff = lat_rsoares - lat_pirulito
lon_diff = lon_rsoares - lon_pirulito
x_diff = (x_r_percent - x_p_percent)*popDensity.shape[1]
y_diff = (y_r_percent - y_p_percent)*popDensity.shape[0]

lat_to_y =  lat_diff/y_diff
lon_to_x = lon_diff/x_diff


test = (-19.933001, -43.938306)

x_test = test[1] - lon_rsoares
y_test = test[0] - lat_rsoares

y_value = y_test/lat_to_y + y_r_percent*popDensity.shape[0]
x_value = x_test/lon_to_x + x_r_percent*popDensity.shape[1]



fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot()


ax1.imshow(popDensity)
ax1.set_axis_off()
ax1.autoscale(False)
ax1.plot(x_value, y_value, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)


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