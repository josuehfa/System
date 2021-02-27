import plotly
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import pi, sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.draw import ellipse 
from skimage.draw import ellipse_perimeter
from skimage.draw import disk
from MapGenClass import *

from ScenarioClass import *
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
                title_text="UAS Path Generation", hovermode="closest",
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

    def animedPlot(self, solution, time_res, costmap, start, goal, region, obstacle,scenario, filename):
        #colorscale to be used in the contour plot
        colorscale=[[0, '#ffffff'],
                    [0.25, '#a0ff7d'],
                    [0.5, '#c5c400'],
                    [0.75, '#dc8000'],
                    [1, '#e10000']]
        colorscale = 'dense'
        #Create the plot Structure
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45,0.45],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map","2D CostMap"),
            specs=[[{"type": "Scattermapbox"},{"type": "contour"}]])
        
        #0
        #Trace for solution  
        fig.append_trace(go.Scattermapbox(
            lat=solution['lat'],
            lon=solution['lon'],
            #lat=[],
            #lon=[],
            name='Solution Path',
            mode='markers+lines',
            marker=dict(size=5, color='blue')
            ),row=1,col=1)

        #1
        #Trace for area of flight
        lat_aux = [reg[0] for reg in region]
        lon_aux = [reg[1] for reg in region]
        fig.append_trace(go.Scattermapbox(
            lon = lon_aux, 
            lat = lat_aux,
            fill = "toself",
            name='Area of Flight',
            marker = { 'size': 5, 'color': "rgba(123, 239, 178, 1)" }),row=1,col=1)

        #2
        #Start position
        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [start[1]],
            lat = [start[0]],
            marker = {'size':5,'color':"black"},
            name='Start'
            ),row=1,col=1)

        #3
        #goal position
        fig.append_trace(go.Scattermapbox(
            mode = "markers",
            lon = [goal[1]],
            lat = [goal[0]],
            marker = {'size':5,'color':"black"},
            name='Goal'
            ),row=1,col=1)

        #4
        #Solution trace for contour
        fig.append_trace(go.Scatter(
            x = solution['lon'], 
            y = solution['lat'],
            #x = [], 
            #y = [],
            mode='markers+lines',
            name='Solution Path',
            marker=dict(size=5, color='red')
            ),row=1,col=2)

        #5
        #Contour plot
        lat,lon = zip(*region)
        lat = list(lat)
        lon = list(lon)
        costmap_y = np.linspace(min(lat),max(lat), len(costmap.y))
        costmap_x = np.linspace(min(lon),max(lon), len(costmap.x))
        fig.append_trace(go.Contour(
            z=costmap.z_time[0],
            x=costmap_x, 
            y=costmap_y, 
            ids=costmap.z_time,
            name='Cost Time: 0',
            line_smoothing=0,
            colorscale=colorscale,
            contours=dict(
                start=costmap.z_time[0].min(), 
                end=costmap.z_time[0].max(), 
                size=1, 
                showlines=False)
            ), row=1, col=2)
        #Fixa o eixo dos plots
        fig.update_xaxes(range=[min(lon),max(lon)],showgrid=False,constrain="domain")
        fig.update_yaxes(range=[min(lat),max(lat)],showgrid=False,constrain="domain")

        #Trace for area of flight
        if obstacle != [] :
            for idx, polygon in enumerate(obstacle):
                lat,lon = zip(*polygon[0])
                lat = list(lat)
                lon = list(lon)
                fig.append_trace(go.Scattermapbox(
                    lon = lon, 
                    lat = lat,
                    fill = "toself",
                    name=polygon[3],
                    marker = { 'size': 5, 'color': "rgba(255, 0, 0, 0)" }),row=1,col=1)
        
        #Cria o perimetro ao redor dos vertiports
        if costmap.verti_perimeters != []:
            vertiport_lat, vertiport_lon = zip(*scenario.vertiports_real)
            fig.append_trace(go.Scattermapbox(
                    lon = vertiport_lon, 
                    lat = vertiport_lat,
                    mode='markers',
                    name='Vertiport',
                    marker=dict(size=5, color='yellow')
                    ),row=1,col=1)
            fig.append_trace(go.Scatter(
                    x = vertiport_lon, 
                    y = vertiport_lat,
                    mode='markers',
                    name='Vertiports',
                    marker=dict(size=5, color='yellow')
                    ),row=1,col=2)
            verti_r_lat = []
            verti_r_lon = []
            for idx, verti_region in enumerate(costmap.verti_perimeters):
                r_lon = verti_region[1]*scenario.lon_range + min(scenario.lon_region)
                r_lat = verti_region[0]*scenario.lat_range + min(scenario.lat_region)

                verti_r_lon.extend(r_lon)
                verti_r_lat.extend(r_lat)
            fig.append_trace(go.Scattermapbox(
                mode = "markers",
                lon = verti_r_lon,
                lat = verti_r_lat,
                marker = {'size':5,'color':"green"},
                name='Vertiport Region'
                ),row=1,col=1)
            fig.append_trace(go.Scatter(
                x = verti_r_lon, 
                y = verti_r_lat,
                mode='markers',
                name='Vertiport Region',
                marker=dict(size=5, color='green')
                ),row=1,col=2)
                
                
                

        #Atualiza o layout
        token = 'pk.eyJ1Ijoiam9zdWVoZmEiLCJhIjoiY2tldnNnODB3MDBtdDJzbXUxMXowMTY5MyJ9.Vwj9BTqB1z9RLKlyh70RHw'  
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
        #for idx, t in enumerate(time_res):
        #    frames.append(go.Frame(data=[go.Scattermapbox(
        #                               lat=solution['lat'][:idx+1], 
        #                               lon=solution['lon'][:idx+1],mode='markers+lines')],traces=[0]))
             
        #    frames.append(go.Frame(data=[go.Scatter(
        #                               x=solution['lon'][:idx], 
        #                               y=solution['lat'][:idx],mode='markers+lines')],traces=[4]))
        #    frames.append(go.Frame(data=[go.Contour(x=costmap_x, y=costmap_y, z=costmap.z_time[t], line_smoothing=0, 
        #                            colorscale=colorscale,
        #                            name='Cost Time: '+str(t),
        #                            contours=dict(
        #                                start=costmap.z_time[t].min(), 
        #                                end=costmap.z_time[t].max(), 
        #                                size=1, 
        #                                showlines=False))],traces=[5]))
        #fig.update(frames=frames)
        plotly.offline.plot(fig, filename=filename,auto_open=False)

        

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
    region = [( 0, 0),
              ( 1, 0),
              ( 1, 1),
              ( 0, 1)]
    solution =[[[10,11,12,13,14,15,16,17,18,19],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[20,21,22,23,24,25,26,27,28,29],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[30,31,32,33,34,35,36,37,38,39],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar'],[[40,41,42,43,44,45,46,47,48,49],
               [10,11,12,13,14,15,16,17,18,19],'RRTstar']]
    final_solution = {"lat":[10,20,30,40],"lon":[10,11,12,13]}
    time_res = [0,1,2,3]

    time_res = [0]
    #Definição dos obstaculos
    obstacle = []
    cpor = [(-19.870782, -43.959432),
            (-19.872528, -43.957651),
            (-19.877101, -43.963053),
            (-19.879381, -43.967742),
            (-19.877050, -43.970242)]
    obstacle.append((cpor,base,topo,'CPOR/CMBH'))

    cdtn = [(-19.871252, -43.969874),
            (-19.872025, -43.969005),
            (-19.870834, -43.966205),
            (-19.871449, -43.965390),
            (-19.872377, -43.965454),
            (-19.874022, -43.967578),
            (-19.874012, -43.967868),
            (-19.874224, -43.967954),
            (-19.874476, -43.967785),
            (-19.875472, -43.969258),
            (-19.875772, -43.971406),
            (-19.872241, -43.971170)]
    obstacle.append((cdtn,base,topo,'CDTN-UFMG'))
    
    mineirao = [(-19.863243, -43.970448),
                (-19.863667, -43.971784),
                (-19.865690, -43.972889),
                (-19.869151, -43.971661),
                (-19.869080, -43.971318),
                (-19.866134, -43.969398)]
    obstacle.append((mineirao,base,topo,'Estadio Mineirão'))

    independencia = [(-19.909092, -43.918879),
                     (-19.909385, -43.916926),
                     (-19.908094, -43.916733),
                     (-19.907786, -43.918641)]
    obstacle.append((independencia,base,topo,'Estadio Independencia'))

    aeroporto_pampulha = [(-19.844565, -43.965362),
                          (-19.847330, -43.950663),
                          (-19.848571, -43.945384),
                          (-19.853395, -43.936264),
                          (-19.857926, -43.936929),
                          (-19.848129, -43.965994)]
    obstacle.append((aeroporto_pampulha,base,topo,'Aeroporto da Pampulha'))

    aeroporto_carlos_prates = [(-19.905402, -43.986880),
                               (-19.910378, -43.993943),
                               (-19.911891, -43.993621),
                               (-19.910698, -43.988835),
                               (-19.911283, -43.988631),
                               (-19.912014, -43.985533),
                               (-19.913010, -43.984980),
                               (-19.912120, -43.983269)]
    obstacle.append((aeroporto_carlos_prates,base,topo,'Aeroporto Carlos Prates'))
    

    start_real = (-19.869245, -43.963622,1) #Escola de Eng
    goal_real = (-19.931071, -43.937778,1) #Praca da liberdade

    start_real = (-19.853342, -44.002499,1) #Zoologico
    goal_real = (-19.927558, -43.911051) #Mangabeiras

    # região da imagem testmap2
    region_real = [(-19.849635, -44.014423), 
              (-19.849635, -43.900210),
              (-19.934877, -43.900210),
              (-19.934877, -44.014423)]

    

    plotSol = PlotlyResult('','','')

    scenario = ScenarioClass('FIVE')

    time = 24
    nrows = 200
    ncols = 200
    vertiports = [(0.062, 0.754), (0.12, 0.94), (0.246, 0.214), (0.602, 0.072), (0.923, 0.234)]
    radius = 0.25
    mapgen = MapGen(nrows, ncols,time)
    #mapgen.create()
    #mapgen.createScenarioTwo()


    final_solution = {"lon": [-43.963655325999994, -43.96305588563015, -43.962480317310245, -43.961884715661995, -43.96131946390639, -43.96179258634359, -43.96133253955185, -43.960885568405544, -43.9602321460132, -43.95953121944105, -43.95885812360049, -43.958620602701494, -43.95855138575264, -43.95768143547004, -43.9571117451926, -43.95664977904005, -43.95610803945424, -43.95570005532419, -43.955921861456645, -43.9556616681082, -43.95564523358134, -43.95560492700454, -43.9553711248671, -43.9550652267396, -43.95465844221005, -43.954165166484444, -43.95346340019194, -43.95289263027404, -43.952286472141395, -43.9518267852298, -43.951700107417, -43.952045712321045, -43.952224932635744, -43.951941227117494, -43.95158998409109, -43.951350783751394, -43.9505933559957, -43.950023305838094, -43.9493277774682, -43.9486660778324, -43.9487200598549, -43.948903958611545, -43.9490844984868, -43.94885885363275, -43.94848793715815, -43.94809542787455, -43.947494188103946, -43.947778613382496, -43.94748099249845, -43.947750422770746, -43.94688119224845, -43.94630250496725, -43.945686150230344, -43.945884084312844, -43.94541144171585, -43.945128815838046, -43.944470715003746, -43.9438849500796, -43.9437496351432, -43.9435003581593, -43.94347432682845, -43.9427402912825, -43.9432570791779, -43.94319937839385, -43.9429063159917, -43.942106302418246, -43.941696758807545, -43.94090262327655, -43.94026323621005, -43.9397309734682, -43.93915936382995, -43.93847799074595, -43.9386254216474, -43.93902860737545, -43.9387036356, -43.93788418849845, -43.93741250558185, -43.93677359835555, -43.93606703366105, -43.9354649541701, -43.93600681371595, -43.935975504142895, -43.936477656912196, -43.936431592253, -43.93647465791095, -43.9362466138559, -43.935492065141396, -43.934864914, -43.93523954923615, -43.9348860269688, -43.93451139173265, -43.93485327787515, -43.93429090516075, -43.934380275398, -43.93366615322035, -43.93385497033905, -43.9332231407557, -43.933305913190196, -43.932592030932646, -43.93204909174635, -43.9313975887148, -43.9308419337632, -43.93023673531095, -43.9296665651933, -43.9292200738872, -43.9288750687834, -43.9285341423213, -43.92789247601385, -43.9272431322632, -43.9267992800782, -43.92671326872235, -43.92611298863215, -43.9255900827742, -43.925275067682904, -43.925344884432, -43.9248501691858, -43.92429367451385, -43.9241112152778, -43.92466555066885, -43.9244279098098, -43.92426488410185, -43.92375577364965, -43.92325697976175, -43.9232350270726, -43.9230445305132, -43.92272339745935, -43.92252642305725, -43.9226289889, -43.9230198187429, -43.9226443437864, -43.9226006783282, -43.922631388101, -43.9226006783282, -43.92258664300235, -43.9223324476564, -43.9220008780782, -43.9218233372042, -43.92110369686425, -43.92084050451455, -43.92083618595275, -43.9208317474309, -43.920687555450804, -43.9199898678, -43.91930909451625, -43.9189215035947, -43.9182461285132, -43.9178848088426, -43.91754400234055, -43.9169479208521, -43.916706081391304, -43.9167024825898, -43.916681249660954, -43.91651354551105, -43.9160165510239, -43.916100163178754, -43.916095844616954, -43.9158469275132, -43.915346094304454, -43.9149136383242, -43.9149090798423, -43.914726380686155, -43.9143171969556, -43.9143121586335, -43.91421151215155, -43.914301362229004, -43.914295364226504, -43.9142888863838, -43.9143906125062, -43.91427461113785, -43.91426669377455, -43.914258176611, -43.9139352441564, -43.9136737312474, -43.913664494323555, -43.9133205688602, -43.91308244816095, -43.913003874328204, -43.9130614551522, -43.91296992563405, -43.912284833788505, -43.9118961632265, -43.911882847660955, -43.911867492774554, -43.91184973868715, -43.911828865638455, -43.9118042738282, -43.911774883615955, -43.911739735321305, -43.9111641670014, -43.9106046733282, -43.910540134821304, -43.9100048730782, -43.9098489250132, -43.9098489250132, -43.9099132236], "lat": [-19.86925224432608, -19.86938064605271, -19.869206675372883, -19.869194326951515, -19.86911717663747, -19.869605868145122, -19.869628488704624, -19.86893184103886, -19.86920153930382, -19.869625210362667, -19.869326334854296, -19.869863764379037, -19.870378136232016, -19.870300876639906, -19.87022744178008, -19.870855025708625, -19.871444690148536, -19.872074241082252, -19.872636804561992, -19.873156203205973, -19.872675379719016, -19.873135331095515, -19.873505565180473, -19.874249967360733, -19.874268544631818, -19.873968029952472, -19.87366489259956, -19.873552117636255, -19.873569711404755, -19.87335388722595, -19.873731333663212, -19.87415063359945, -19.874824551427647, -19.874362086655644, -19.874821819476015, -19.875293900717754, -19.87502802718508, -19.87520024941586, -19.875623920474712, -19.875296632669386, -19.874828157603798, -19.875324717132145, -19.874985408739647, -19.87441934836182, -19.874802367980408, -19.87500923135786, -19.874645444678755, -19.87528876464869, -19.875512566126254, -19.875850235347777, -19.875777456156342, -19.87571877383532, -19.876073927547274, -19.87642547508308, -19.875967927824014, -19.876169764410474, -19.875824008612124, -19.87606835436595, -19.875381432447995, -19.87580772618041, -19.876429190537298, -19.876195226199666, -19.876535299538624, -19.8759211568121, -19.87547836209184, -19.875166810327908, -19.874938747005796, -19.874630036471558, -19.87483515139997, -19.875077530148623, -19.87452917281736, -19.87416046862532, -19.873752861442057, -19.87391645070569, -19.873295532739125, -19.873327004821906, -19.872826402005145, -19.872325471354188, -19.872043096833668, -19.872077410146144, -19.871859618962166, -19.87155309398923, -19.87194321668206, -19.87158992069721, -19.87102877783232, -19.87139278306756, -19.871384150100408, -19.871547083695646, -19.871973705262253, -19.871949664087907, -19.872534848127145, -19.872747066129797, -19.872798317542383, -19.873176091813843, -19.873215759751513, -19.87370456053723, -19.873363394417623, -19.873686857490668, -19.87406812866021, -19.874672108526667, -19.875166701049842, -19.875758223216863, -19.876299477473886, -19.876871766701427, -19.877509076377777, -19.877467550712993, -19.878011974033907, -19.8785303891753, -19.879036565173386, -19.878707856753213, -19.87909109492793, -19.87963715741982, -19.88022714969393, -19.880206277583472, -19.88083123883845, -19.881227590380995, -19.88180873113182, -19.882473469602537, -19.8825266880203, -19.882975493034145, -19.883425609384776, -19.884033851095776, -19.884625263984734, -19.88516367701206, -19.88559324908643, -19.886283340068278, -19.886748208957712, -19.887283125086952, -19.887921199709755, -19.888445406588602, -19.888973875311994, -19.889533269727842, -19.89006665596417, -19.89060648960634, -19.890915090862517, -19.89115943661634, -19.89158267056293, -19.891945364461385, -19.892268499700233, -19.892813250855344, -19.893357892732386, -19.89382396368054, -19.89362125286956, -19.893596446748756, -19.893847349186494, -19.893562679826605, -19.89392220466117, -19.89368747537708, -19.89421692760306, -19.89447449600278, -19.895019902826277, -19.895560173580712, -19.896027009475322, -19.8966293501708, -19.897204808462234, -19.897749996729605, -19.89816022658643, -19.89877928682589, -19.899393757386605, -19.899938945653975, -19.900406218660844, -19.901033693311323, -19.901578881578693, -19.902090849314234, -19.90266882100117, -19.90321357215628, -19.90375810475526, -19.904325476469864, -19.904846514284824, -19.90539039121541, -19.905933831033735, -19.90621401999295, -19.90648535742889, -19.907028797247214, -19.906847177102822, -19.907580760754627, -19.90809753672504, -19.908666875444823, -19.909173597833234, -19.909665130570584, -19.91031664639541, -19.91085910271115, -19.911400684802366, -19.911940846278736, -19.912479259306064, -19.913015049659823, -19.91354712455937, -19.91407373555565, -19.91463400419602, -19.91520061096417, -19.915712906533912, -19.916293391616346, -19.916737497673388, -19.916737497673388, -19.917006758826084]}
    time_res = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19]
    scenario = ScenarioClass('FIVE')
    plotSol.animedPlot(final_solution, time_res, scenario.mapgen, scenario.start_real, scenario.goal_real, scenario.region_real, scenario.obstacle, scenario,'6M.html')


    #start =(0.1,0.1,1) 
    #goal = (0.9,0.9,1)
    #region = [(0, 0), (1, 0), (1, 1), (0, 1)]
    #time_res = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #path_x = [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9]
    #path_y = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    #final_solution = {"lon":path_x,"lat":path_y}

    #plotSol.simplePlot(solution, final_solution,obstacle,'costmap',start,goal,region)

    #time_res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]