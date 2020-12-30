import numpy as np
from numpy import sin, cos, pi
import json
from IPython.display import display, HTML
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
init_notebook_mode(connected=True)
init_notebook_mode(connected=True)

def get_sliders(n_frames, fr_duration=100, x_pos=0.0, slider_len=1.0):
    # n_frames= number of frames
    #fr_duration=the duration in milliseconds of each frame
    #x_pos x-coordinate where the slider starts
    #slider_len is a number in (0,1] giving the slider length as a fraction of x-axis length 
    return [dict(steps= [dict(method= 'animate',#Sets the Plotly method to be called when the slider value is changed.
                              args= [ [ 'frame{}'.format(k) ],#Sets the arguments values to be passed to the Plotly,
                                                              #method set in method on slide
                                      dict(mode= 'immediate',
                                           frame= dict( duration=fr_duration, redraw= True ),
                                           transition=dict( duration= 0)
                                          )
                                    ],
                              label='fr{}'.format(k)
                             ) for k in range(n_frames)], 
                transition= { 'duration': 0 },
                x=x_pos,
                len=slider_len)]

def get_updatemenus(x_pos=0.0, fr_duration=200):
    return [dict(x= x_pos,
                 y= 0,
                 yanchor='top',
                 xanchor= 'right',
                 pad= dict(r= 10, t=40 ),
                 type='buttons',
                 showactive= False,
                 buttons= [dict(label='Play',
                                method='animate',
                                args= [ None,
                                        dict(mode='immediate',
                                             transition= { 'duration': 0 },
                                             fromcurrent= True,
                                             frame= dict( redraw=True, duration=fr_duration)
                                            )
                                       ]
                                ),
                           dict(label='Pause',
                                method='animate',
                                args= [ [None],
                                        dict(mode='immediate',
                                             transition= { 'duration': 0 },
                                             frame= dict( redraw=True, duration=0 )
                                            )
                                       ]
                                )
                           ]
               )
        ]

pl_ocean=[[0, '#193f6e'],
[0.25, '#3b6ba5'],
[0.5, '#72a5d3'],
[0.75, '#b1d3e3'],
[1, '#e1ebec']]

t=np.linspace(0, 4, 25)
X=np.linspace(-1,1, 100)
Y=np.linspace(-1,1, 100)
x,y=np.meshgrid(X,Y)
N=t.shape[0]-1
r=np.sqrt(x**2+y**2)
z=cos(4*pi*r)*cos(pi*(r-t[0]*pi/3))
data=[dict(type='contour', 
           x=X,
           y=Y,
           z=z,
           colorscale=pl_ocean,
           contours=dict(start=np.min(z), end=np.max(z), size=0.2, showlines=False)
           )
    ]

layout=dict(width=600, height=600,
            font=dict(family='Balto', 
                      size=12),
            
            xaxis= dict(range= [-1,1], 
                        ticklen=4,  
                        autorange= False, 
                        zeroline=False, 
                        showline=True, 
                        mirror=True,
                        showgrid=False),
            yaxis=dict(range= [-1, 1], 
                       ticklen=4,  
                       autorange= False, 
                       showline=True, 
                       mirror=True,
                       zeroline=False, 
                       showgrid=False),
            title= 'Animating the contour of wave', 
            hovermode='closest',
            sliders=get_sliders(n_frames=N),
            updatemenus=get_updatemenus()
            )

   
frames=[{"data": [ {'z': cos(4*pi*r)*cos(pi*(r-t[k]*pi/3))} ],
             "name":'frame{}'.format(k-1) ,  
             "traces": [0]
            } for k in range(1, len(t)) ]
            
fig=dict(data=data, layout=layout, frames=frames)

#fig.show()

iplot(fig, validate=False)