import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.draw import ellipse 
from skimage.draw import ellipse_perimeter
from skimage.draw import disk

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
            column_widths=[0.5,0.5],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map","2D CostMap"),
            specs=[[{"type": "Scattermapbox"},{"type": "contour"}]])
        
        #fig.append_trace(go.Scattermapbox(
        #    mode = "markers+text",
        #    lon = [start[1],goal[1]],
        #    lat = [start[0],goal[0]],
        #    marker = {'size':15,'symbol':["marker","marker"]},
        #    name=' ',
        #    text = ["Start", "Goal"],textposition = "bottom center"),row=1,col=1)

        #for obs in obstacle:
        #    lat_aux_1 = [obs_[0] for obs_ in obs[0]]
        #    lon_aux_1 = [obs_[1] for obs_ in obs[0]]
        #    fig.add_trace(go.Scattermapbox(
        #        fill = "toself",
        #        lon = lon_aux_1,
        #        lat = lat_aux_1,
        #        name='Obstacle(CB)',
        #        marker = {'size': 2, 'color': "rgba(108, 122, 137, 1)"}))

        #lat_aux = [reg[0] for reg in region]
        #lon_aux = [reg[1] for reg in region]
        #fig.add_trace(go.Scattermapbox(
        #    lon = lon_aux, 
        #    lat = lat_aux,
        #    fill = "toself",
        #    name='Area of Flight',
        #    marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }))
        
        
    
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

        y=np.arange(100) 
        #fig.append_trace(go.Scatter(x=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5], y=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3], marker_size = 100,mode='markers',marker_symbol='square', marker=dict(
        #            color=z,
        #            line_width=1)),row=1,col=1)

        #filename = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
        #df = pd.read_csv(filename, encoding='utf-8')
        #df = df.head(100)
        lat=np.arange(100)
        lon=np.arange(100)
        fig.append_trace(go.Scattermapbox(
               lat=lat,
               lon=lon,
               mode='markers+lines',
               marker=dict(size=10, color='red')
            ),row=1,col=1)

        fig.append_trace(go.Contour(x=X, y=Y, z=z_t[0], ids=z_t,
                                    name='testcolormap',
                                    line_smoothing=0, 
                                    colorscale=colorscale,
                                    contours=dict(
                                        start=np.min(z), 
                                        end=np.max(z), 
                                        size=0.5, 
                                        showlines=False)
                                    ), row=1, col=2)


        fig.update_layout(
                mapbox = {
                    'style': "stamen-terrain",
                    'center': {'lon': -73, 'lat': 46 },
                    'zoom': 5},
                title_text="Generation of a Path for a UAS", hovermode="closest",
                updatemenus=[dict(  x=0,
                                    y=0,
                                    yanchor='top',
                                    xanchor= 'right',
                                    pad= dict(r= 10, t=40 ),
                                    type="buttons",
                                    showactive= False,
                                    buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, dict(frame= { "duration": 50},
                                                                fromcurrent= True,
                                                                mode='immediate', 
                                                                transition= {"duration": 50, "easing": "linear"})]),
                                            dict(label='Pause',
                                                method='animate',
                                                args= [ [None],
                                                        dict(frame= { "duration": 0},
                                                                fromcurrent= True,
                                                                mode='immediate', 
                                                                transition= {"duration": 0, "easing": "linear"})
                                       ]
                                )])],
                sliders=get_sliders(n_frames=100)
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
        for i in range(1, 100):
            frames.append(go.Frame(data=[go.Contour(x=X, y=Y, z=z_t[move], line_smoothing=0, colorscale=colorscale)],traces=[1]))
            frames.append(go.Frame(data=[go.Scattermapbox(
                                       lat=lat[:i], 
                                       lon=lon[:i],mode='markers+lines')],traces=[0]))
            if i%10 == 0 and i < 100:
                move = move + 1
                
        #frames= [go.Frame(data=[go.Scatter(y=y[:i],name='Testing Points'),go.Contour(z=z, line_smoothing=0, colorscale='dense',name='testcolormap')]) for i in range(1, 100)]



        fig.update(frames=frames)
        
        fig.show()
        print()

    def animedPlot(self, solution, obstacle, costmap):
        #Generate a animed plot with plotly
        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.2, 0.2, 0.6],
            subplot_titles=("3D Path","Altitude vs Latitude", "Altitude vs Longitude ", "Paths vs Map"),
            specs=[[{"type": "mesh3d", "rowspan": 3}, {"type": "scatter"}],
                    [        None     , {"type": "scatter"}],
                    [        None     , {"type": "scattermapbox"}]])

        #fig_aux = px.line_3d(self.solutionDataFrame, x="latitude", y="longitude", z="altitude",color='algorithm')
        
        fig_aux = px.line_3d(self.solutionData[0][1], x="latitude", y="longitude", z="altitude",color='algorithm')
        
        fig.append_trace(fig_aux['data'][0],row=1,col=1)
        for polygon in self.obstacle:
            lat_pnt = []
            lon_pnt = []
            alt_pnt = []
            for point in polygon[0]:
                lat_pnt.append(point[0])
                lon_pnt.append(point[1])
                alt_pnt.append(polygon[1])
            for point in polygon[0]:
                lat_pnt.append(point[0])
                lon_pnt.append(point[1])
                alt_pnt.append(polygon[2])
        
            fig.add_trace(go.Mesh3d(x=lat_pnt,
                y=lon_pnt,
                z=alt_pnt,
                color='rgba(108, 122, 137, 1)',
                colorbar_title='z',
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                name=polygon[3],
                showscale=True
                ))
        fig.update_scenes(xaxis_autorange="reversed")
        
        fig.add_trace(go.Scatter3d(
            x=[self.start[0],self.goal[0]],
            y=[self.start[1],self.goal[1]],
            z=[self.start[2],self.goal[2]],
            mode='markers+text',
            name=' ',
            text=['Start','Goal'],
            textposition = "bottom center",
            marker=dict(
                size=5,
                color="rgba(30, 130, 76, 1)",
                opacity=0.8
            )))
        

        #fig_aux_2 = px.line(self.solutionDataFrame, x="latitude", y="altitude", title='Latitude vs Altitude',color='algorithm')
        #fig.append_trace(fig_aux_2['data'][0],row=1,col=2)

        #fig_aux_3 = px.line(self.solutionDataFrame, x="longitude", y="altitude", title='Longitude vs Altitude',color='algorithm')
        #fig.append_trace(fig_aux_3['data'][0],row=2,col=2)

        fig_aux_2 = px.line(self.solutionData[0][1], x="latitude", y="altitude", title='Latitude vs Altitude',color='algorithm')
        fig.append_trace(fig_aux_2['data'][0],row=1,col=2)

        fig_aux_3 = px.line(self.solutionData[0][1], x="longitude", y="altitude", title='Longitude vs Altitude',color='algorithm')
        fig.append_trace(fig_aux_3['data'][0],row=2,col=2)


        lat_aux = [reg[0] for reg in self.region]
        lon_aux = [reg[1] for reg in self.region]
        fig.add_trace(go.Scattermapbox(
            fill = "toself",
            lon = lon_aux, 
            lat = lat_aux,
            name='Area of Flight',
            marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }),row=3,col=2)
        
        for obstacle in self.obstacle:
            lat_aux_1 = [obs_[0] for obs_ in obstacle[0]]
            lon_aux_1 = [obs_[1] for obs_ in obstacle[0]]
            fig.add_trace(go.Scattermapbox(
                fill = "toself",
                lon = lon_aux_1,
                lat = lat_aux_1,
                name='Obstacle(CB)',
                marker = {'size': 2, 'color': "rgba(108, 122, 137, 1)"}))
        
        for path in self.solution:
            fig.add_trace(go.Scattermapbox(
                mode = "markers+lines",
                lon = path[1],
                lat = path[0],
                marker = {'size': 5},
                name=path[3]))

        fig.add_trace(go.Scattermapbox(
                mode = "markers+text",
                lon = [self.start[1],self.goal[1]],
                lat = [self.start[0],self.goal[0]],
                marker = {'size':15,'symbol':["marker","marker"]},
                name=' ',
                text = ["Start", "Goal"],textposition = "bottom center"))
        
        token = 'pk.eyJ1Ijoiam9zdWVoZmEiLCJhIjoiY2tldnNnODB3MDBtdDJzbXUxMXowMTY5MyJ9.Vwj9BTqB1z9RLKlyh70RHw'
        
        fig.update_layout(
            mapbox = {
                'style': "outdoors",
                'center': {'lon': -47.5, 'lat': -12.3 },
                'accesstoken': token,
                'zoom': 7
            },
            showlegend = False)

        fig.update_geos(
            projection_type="orthographic",
            landcolor="white",
            oceancolor="MidnightBlue",
            showocean=True,
            lakecolor="LightBlue"
        )

        fig.show()


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
    final_solution = [[10,20,30,40],[10,11,12,13]]
    plotSol = PlotlyResult('','','')
    plotSol.simplePlot(solution, final_solution,obstacle,'costmap',start,goal,region)