import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.draw import ellipse 
from skimage.draw import ellipse_perimeter
from skimage.draw import disk
from MapGenClass import *
class PlotlyResult():
#Class to create plots using Plotly Library

    def __init__(self, solution, obstacle, costmap):
        self.solution = solution
        self.obstacle = obstacle
        self.map = costmap
    
    def simplePlot(self, solution, final_solution, obstacle, costmap, start, goal, region):
        #Generate a simple solution plot using plotly

        t = np.linspace(-1, 1, 100)
        x = t + t ** 2
        y = t - t ** 2
        xm = np.min(x) - 1.5
        xM = np.max(x) + 1.5
        ym = np.min(y) - 1.5
        yM = np.max(y) + 1.5
        N = 200
        s = np.linspace(-1, 1, N)
        xx = s + s ** 2
        yy = s - s ** 2
        

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



        #Create a Plot Structure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45,0.45],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map","2D CostMap"),
            specs=[[{"type": "Scattermapbox"},{"type": "contour"}]])
        
            
    
        #data_fig.append(go.Scattermapbox(
        #    mode = "markers+lines",
        #    lon = [],
        #    lat = [],
        #    marker = {'size': 5},
        #    name='test'))

        #data_fig.append(go.Scattermapbox(
        #    mode = "markers+lines",
        #    lon = [],
        #    lat = [],
        #    marker = {'size': 5},
        #    name='FinalResult'))
        

        
        z =    [[2, 4, 7, 12, 13, 14, 15, 16],
                [3, 1, 6, 11, 12, 13, 16, 17],
                [4, 2, 7, 7, 11, 14, 17, 18],
                [5, 3, 8, 8, 13, 15, 18, 19],
                [7, 4, 10, 9, 16, 18, 20, 19],
                [9, 10, 5, 27, 23, 21, 21, 21],
                [11, 14, 17, 26, 25, 24, 23, 22]]
        
        nrows = 80
        ncols = 80
        time = 10
        z_t = []
        for t in range(time):
            z = np.zeros((nrows+1, ncols+1), dtype=np.uint8) + 1
            xx,yy = disk((0.5*nrows,0.5*nrows),0.5*nrows)
            z[xx,yy] = 10

            xx,yy = disk((0.5*nrows,0.5*nrows),0.25*nrows)
            z[xx,yy] = 20

            xx,yy = disk((0.5*nrows,0.5*nrows),0.125*nrows)
            z[xx,yy] = 50

            #Region Clearece
            xx, yy = ellipse(0.5*nrows+t*(0.5*nrows/time), 0.5*ncols-t*(0.5*ncols/time), 0.15*nrows, 0.25*ncols, rotation=np.deg2rad(10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            z[xx,yy] = 50

                #Region Clearece
            xx, yy = ellipse(0.5*nrows-t*(0.5*nrows/time), 0.5*ncols+t*(0.5*ncols/time), 0.15*nrows, 0.25*ncols, rotation=np.deg2rad(10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            z[xx,yy] = 50

            z = np.asarray(z,dtype=np.double)
            z_t.append(z)

        X=np.linspace(-1,1, 80)
        Y=np.linspace(-1,1, 80)
        colorscale=[[0, '#e1ebec'],
                    [0.25, '#b1d3e3'],
                    [0.5, '#72a5d3'],
                    [0.75, '#3b6ba5'],
                    [1, '#193f6e']]
        colorscale=[[0, '#ffffff'],
                    [0.25, '#a0ff7d'],
                    [0.5, '#c5c400'],
                    [0.75, '#dc8000'],
                    [1, '#e10000']]

        y=np.arange(100) 
        #fig.append_trace(go.Scatter(x=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5], y=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3], marker_size = 100,mode='markers',marker_symbol='square', marker=dict(
        #            color=z,
        #            line_width=1)),row=1,col=1)

        #filename = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
        #df = pd.read_csv(filename, encoding='utf-8')
        #df = df.head(100)
        lat=np.arange(50)
        lon=np.arange(50)
        fig.append_trace(go.Scattermapbox(
               lat=lat,
               lon=lon,
               mode='markers+lines',
               marker=dict(size=5, color='red')
            ),row=1,col=1)

        lat_aux = [reg[0] for reg in region]
        lon_aux = [reg[1] for reg in region]
        fig.append_trace(go.Scattermapbox(
            lon = lon_aux, 
            lat = lat_aux,
            fill = "toself",
            name='Area of Flight',
            marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }),row=1,col=1)

        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [start[1]],
            lat = [start[0]],
            marker = {'size':5,'color':"black"},
            name='Start'
            ),row=1,col=1)

        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [goal[1]],
            lat = [goal[0]],
            marker = {'size':5,'color':"black"},
            name='Goal'
            ),row=1,col=1)

        fig.append_trace(go.Scatter(
            x = [], 
            y = [],
            ids=X,
            mode='markers+lines',
            marker=dict(size=5, color='black')
            ),row=1,col=2)
        fig.append_trace(go.Contour(x=X, y=Y, z=z_t[0], ids=z_t,
                                    name='testcolormap',
                                    line_smoothing=0, 
                                    colorscale=colorscale,
                                    contours=dict(
                                        start=0, 
                                        end=50, 
                                        size=1, 
                                        showlines=False)
                                    ), row=1, col=2)


        fig.update_xaxes(range=[-1,1],showgrid=False,constrain="domain")
        fig.update_yaxes(range=[-1,1],showgrid=False,constrain="domain")
        fig.update_layout(
                mapbox = {
                    'style': "stamen-terrain",
                    'center': {'lon': 0, 'lat': 0 },
                    'zoom': 5},
                title_text="Generation of a Path for a UAS", hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    #y=1.02,
                    xanchor="right",
                    x=1
                ),
                updatemenus=[dict(  #x=0,
                                    #y=0,
                                    #yanchor='top',
                                    #xanchor= 'right',
                                    #pad= dict(r= 10, t=40 ),
                                    type="buttons",
                                    showactive= False,
                                    buttons=[dict(label="Play",
                                            method="animate",
                                            args=[[None], dict(frame= { "duration": 50},
                                                                fromcurrent= True,
                                                                mode='immediate'#, 
                                                                #transition= {"duration":10, "easing": "linear"}
                                                                )]),
                                            dict(label='Pause',
                                                method='animate',
                                                args= [ [None],
                                                        dict(frame= { "duration": 0},
                                                                fromcurrent= True,
                                                                mode='immediate'#, 
                                                                #transition= {"duration": 0, "easing": "linear"}
                                                                )
                                      ]
                                )])]#,
                #sliders=get_sliders(n_frames=100)
                            )

        #frames = [go.Frame(
        #            data=[go.Scattermapbox(
        #                lon=solution[k][1],
        #                lat=solution[k][0],
        #                mode="markers",
        #                marker=dict(color="orange", size=10),
        #                name='test'),
        #                go.Scattermapbox(
        #                lon=final_solution[1][:k],
        #                lat=final_solution[1][:k],
        #                mode="lines",
        #                marker=dict(color="red", size=10),
        #                name='FinalResult')]) for k in range(len(solution))]

        move = 0
        frames = []
        
        for i in range(1, 50):
            frames.append(go.Frame(data=[go.Scattermapbox(
                                       lat=lat[:i], 
                                       lon=lon[:i],mode='markers+lines')],traces=[0]))
             
            frames.append(go.Frame(data=[go.Scatter(
                                       x=X[:i], 
                                       y=Y[:i],mode='markers+lines')],traces=[4]))
            frames.append(go.Frame(data=[go.Contour(x=X, y=Y, z=z_t[move], line_smoothing=0, 
                                    colorscale=colorscale,
                                    contours=dict(
                                        start=0, 
                                        end=50, 
                                        size=1, 
                                        showlines=False))],traces=[5]))
            
            if i%10 == 0 and i < 100:
                move = move + 1
                
        #frames= [go.Frame(data=[go.Scatter(y=y[:i],name='Testing Points'),go.Contour(z=z, line_smoothing=0, colorscale='dense',name='testcolormap')]) for i in range(1, 100)]



        fig.update(frames=frames)
        
        plotly.offline.plot(fig, filename='path.html')
        #fig.show()
        print()

    def animedPlot(self, solution, time_res, costmap, start, goal, region):
        #colorscale to be used in the contour plot
        colorscale=[[0, '#ffffff'],
                    [0.25, '#a0ff7d'],
                    [0.5, '#c5c400'],
                    [0.75, '#dc8000'],
                    [1, '#e10000']]
        
        #Create the plot Structure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45,0.45],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map","2D CostMap"),
            specs=[[{"type": "Scattermapbox"},{"type": "contour"}]])
        
        #Trace for solution
        fig.append_trace(go.Scattermapbox(
            lat=[],
            lon=[],
            name='Solution Path',
            mode='markers+lines',
            marker=dict(size=5, color='red')
            ),row=1,col=1)

        #Trace for area of flight
        lat_aux = [reg[0] for reg in region]
        lon_aux = [reg[1] for reg in region]
        fig.append_trace(go.Scattermapbox(
            lon = lon_aux, 
            lat = lat_aux,
            fill = "toself",
            name='Area of Flight',
            marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }),row=1,col=1)

        #Start position
        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [start[1]],
            lat = [start[0]],
            marker = {'size':5,'color':"black"},
            name='Start'
            ),row=1,col=1)

        #goal position
        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [goal[1]],
            lat = [goal[0]],
            marker = {'size':5,'color':"black"},
            name='Goal'
            ),row=1,col=1)

        #Solution trace for contour
        fig.append_trace(go.Scatter(
            x = [], 
            y = [],
            mode='markers+lines',
            name='Solution Path',
            marker=dict(size=5, color='black')
            ),row=1,col=2)

        #Contour plot
        lat,lon = zip(*region)
        lat = list(lat)
        lon = list(lon)
        costmap_x = np.linspace(lat[0],lat[1], len(costmap.x))
        costmap_y = np.linspace(lon[0],lon[1], len(costmap.y))
        fig.append_trace(go.Contour(
            x=costmap_x, 
            y=costmap_y, 
            z=costmap.z_time[0], 
            ids=costmap.z_time,
            name='Cost',
            line_smoothing=0,
            colorscale=colorscale,
            contours=dict(
            start=costmap.z_time[0].min(), 
            end=costmap.z_time[0].max(), 
            size=1, 
            showlines=False)
            ), row=1, col=2)
        #Fixa o eixo dos plots
        fig.update_xaxes(range=[lat[0],lat[1]],showgrid=False,constrain="domain")
        fig.update_yaxes(range=[lon[0],lon[1]],showgrid=False,constrain="domain")
        #Atualiza o layout
        fig.update_layout(
            mapbox = {
                'style': "stamen-terrain",
                'center': {'lon': 0, 'lat': 0 },
                'zoom': 5},
            title_text="Generation of a Path for a UAS", hovermode="closest",
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
                                mode='immediate'#, 
                                #transition= {"duration":10, "easing": "linear"}
                                )]),
                    dict(label='Pause',
                        method='animate',
                        args= [ [None],
                            dict(frame= { "duration": 0},
                                fromcurrent= True,
                                mode='immediate'#, 
                                #transition= {"duration": 0, "easing": "linear"}
                                        )]
                    )]
                )])

        frames = []
        for idx, t in enumerate(time_res):
            frames.append(go.Frame(data=[go.Scattermapbox(
                                       lat=solution['lat'][:idx], 
                                       lon=solution['lon'][:idx],mode='markers+lines')],traces=[0]))
             
            #frames.append(go.Frame(data=[go.Scatter(
            #                           x=solution['lat'][:idx], 
            #                           y=solution['lon'][:idx],mode='markers+lines')],traces=[4]))
            frames.append(go.Frame(data=[go.Contour(x=costmap_x, y=costmap_y, z=costmap.z_time[t], line_smoothing=0, 
                                    colorscale=colorscale,
                                    contours=dict(
                                        start=costmap.z_time[t].min(), 
                                        end=costmap.z_time[t].max(), 
                                        size=1, 
                                        showlines=False))],traces=[5]))
        fig.update(frames=frames)
        plotly.offline.plot(fig, filename='path.html')



if __name__ == "__main__":

    #Obstacle Sample
    base = 2
    topo = 6
    type_obstacle = 'CB' 
    polygon = [(-3,-2),
               (-3, 5), 
               ( 6, 5),
               ( 6,-2)]
    polygon2 = [(8, 9),
                (8, 6),
                (6, 6),
                (6, 9)]
    obstacle = [(polygon, base, topo, type_obstacle),(polygon2, base, topo,type_obstacle)]
    start =(-9,-9,1) 
    goal = (9,9,8)
    region = [( 10, 10),
              ( 30,-10),
              ( 20,-20),
              (-10,-20),
              (-30, 0)]
    solution =[[[10,11,12,13,14,15,16,17,18,19],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[20,21,22,23,24,25,26,27,28,29],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[30,31,32,33,34,35,36,37,38,39],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[40,41,42,43,44,45,46,47,48,49],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar']]
    final_solution = {"lat":[10,20,30,40],"lon":[10,11,12,13]}
    plotSol = PlotlyResult('','','')
    #plotSol.simplePlot(solution, final_solution,obstacle,'costmap',start,goal,region)

    time = 4
    nrows = 100
    ncols = 100
    mapgen = MapGen(nrows, ncols,time)
    mapgen.create()
    time_res = [0,1,2,3]
    plotSol.animedPlot(final_solution, time_res, mapgen, start, goal, region)