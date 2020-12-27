import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotlyResult():
#Class to create plots using Plotly Library

    def __init__(self, solution, obstacle, costmap):
        self.solution = solution
        self.obstacle = obstacle
        self.map = costmap
    
    def simplePlot(self, solution, obstacle, costmap, start, goal, region):
        #Generate a simple solution plot using plotly

        t = np.linspace(-1, 1, 100)
        x = t + t ** 2
        y = t - t ** 2
        xm = np.min(x) - 1.5
        xM = np.max(x) + 1.5
        ym = np.min(y) - 1.5
        yM = np.max(y) + 1.5
        N = 50
        s = np.linspace(-1, 1, N)
        xx = s + s ** 2
        yy = s - s ** 2
        
        #Create a Plot Structure
        fig = make_subplots(
            rows=1, cols=1,
            column_widths=[1],
            row_heights=[1],
            subplot_titles=("Solution Path in a Real Map"),
            specs=[[{"type": "scattermapbox"}]])
        
        # Create figure
        lat_aux = [reg[0] for reg in region]
        lon_aux = [reg[1] for reg in region]
        fig = go.Figure(
            data=[go.Scatter(x=x, y=y,
                            mode="lines",
                            line=dict(width=2, color="blue")),
                go.Scatter(x=x, y=y,
                            mode="lines",
                            line=dict(width=2, color="blue"))],
            layout=go.Layout(
                xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
                updatemenus=[dict(type="buttons",
                                buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None])])]),
            frames=[go.Frame(
                data=[go.Scatter(
                    x=[xx[k]],
                    y=[yy[k]],
                    mode="markers",
                    marker=dict(color="red", size=10))])

                for k in range(N)]
        )

        
        
        #for obstacle in self.obstacle:
        #    lat_aux_1 = [obs_[0] for obs_ in obstacle[0]]
        #    lon_aux_1 = [obs_[1] for obs_ in obstacle[0]]
        #    fig.add_trace(go.Scattermapbox(
        #        fill = "toself",
        #        lon = lon_aux_1,
        #        lat = lat_aux_1,
        #        name='Obstacle(CB)',
        #        marker = {'size': 2, 'color': "rgba(108, 122, 137, 1)"}))
        
        #for path in self.solution:
        #    fig.add_trace(go.Scattermapbox(
        #        mode = "markers+lines",
        #        lon = path[1],
        #        lat = path[0],
        #        marker = {'size': 5},
        #        name=path[3]))

        #fig.add_trace(go.Scattermapbox(
        #        mode = "markers+text",
        #        lon = [start[1],goal[1]],
        #        lat = [start[0],goal[0]],
        #        marker = {'size':15,'symbol':["marker","marker"]},
        #        name=' ',
        #        text = ["Start", "Goal"],textposition = "bottom center"))
        
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
    start =(-10,-10,1) 
    goal = (10,10,8)
    region = [( 10, 10),
              ( 30,-10),
              ( 20,-20),
              (-10,-20),
              (-30, 0)]
    plotSol = PlotlyResult('','','')
    plotSol.simplePlot('solution',obstacle,'costmap',start,goal,region)