import os
import numpy as np
from skimage import io
from skimage.util import crop
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.transform import rotate, resize


nrows = 200
ncols = 200
delta_d = 1/nrows
x = np.arange(ncols+1)*delta_d
y = np.arange(nrows+1)*delta_d


mypath = '/home/josuehfa/System/CoreSystem/MeteorologicalData/'
(_, _, filenames) = next(os.walk(mypath))
filenames = sorted(filenames)
time = len(filenames)

z_time = []
for t in range(time):

    original = io.imread(os.path.join(mypath, filenames[t]))
    image_resized = rgb2gray(resize(original, (2*nrows+1,2*ncols+1),anti_aliasing=True))
    image_croped = crop(image_resized, ((nrows/2, nrows/2), (nrows/2, nrows/2)), copy=False)
    image_rotate = rotate(image_croped,180)
    final_image = image_rotate[:,::-1]
    final_image = np.multiply(final_image, np.where(final_image >= 0.1, 110, 1))
    z_time.append(final_image)

    #import matplotlib.pyplot as plt
    #fig,ax0 = plt.subplots(ncols=1, figsize=(8, 2))
    #ax0.imshow(final_image)
    #ax0.axis('off')
    #fig.tight_layout()
    #plt.show()




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot

ims = []
fig = plt.figure()
axis = plt.axes(xlim =(0, 1),  
                ylim =(0, 1))
for t in range(time):
        
    aux_im = []
    aux_im.append(axis.pcolormesh(x, y, z_time[t]*delta_d, cmap='rainbow', shading='nearest'))
    test = tuple(aux_im)
    ims.append(test)
    pyplot.clf()

    
  
im_ani = animation.ArtistAnimation(fig, ims, interval=10000/time, blit=True)

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
x = np.linspace(min(lon_aux),max(lon_aux),nrows)
y = np.linspace(min(lat_aux),max(lat_aux),nrows)
fig.append_trace(go.Contour(
    z=z_time[0],
    x=x, 
    y=y,
    name='Cost Time: 0',
    line_smoothing=0,
    colorscale='dense',
    contours=dict(
        start=z_time[0].min(), 
        end=z_time[0].max(), 
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
                    dict(frame= { "duration": 100},
                        fromcurrent= True,
                        mode='immediate', 
                        transition= {"duration":0, "easing": "linear"}
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
frames = []
colorscale = 'dense'
for t in range(time):
    #frames.append(go.Frame(data=[go.Scattermapbox(
    #                           lat=solution['lat'][:idx+1], 
    #                           lon=solution['lon'][:idx+1],mode='markers+lines')],traces=[0]))
        
    #frames.append(go.Frame(data=[go.Scatter(
    #                           x=solution['lon'][:idx], 
    #                           y=solution['lat'][:idx],mode='markers+lines')],traces=[4]))
    frames.append(go.Frame(data=[go.Contour(x=x, y=y, z=z_time[t], line_smoothing=0, 
                            colorscale=colorscale,
                            name='Cost Time: '+str(t),
                            contours=dict(
                                start=z_time[0].min(), 
                                end=z_time[0].max(), 
                                size=1, 
                                showlines=False))],traces=[2]))
fig.update(frames=frames)
plotly.offline.plot(fig, filename='test.html')
