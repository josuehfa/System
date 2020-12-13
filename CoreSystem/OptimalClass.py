import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
from math import sqrt
import argparse
from shapely.geometry import Point,LineString
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Ellipse
import matplotlib as mpl
mpl.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.draw import line_aa
from euclid import *

class OptimalPlanning():
    def __init__(self, start, goal, region, obstacle, planner, dimension):
        self.start = start
        self.goal = goal
        self.region = region
        self.obstacle = obstacle
        self.planner = planner
        self.dimension = dimension
        self.solution = []
        self.solutionSampled = []
        self.PlannerStates = []

    def plan(self,runTime, plannerType, objectiveType):
        if self.planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar']:
            result = self.OMPL_plan(runTime, plannerType, objectiveType)
            return result
        else:
            return False
            pass
    

    class ValidityChecker(ob.StateValidityChecker):
        #Obstable structure: [(polygon,base,topo),(polygon,base,topo)]
        def obstacle(self,start, goal, region, obstacle,dimension):
            self.start = start 
            self.goal = goal
            self.region = region
            self.obstacle = obstacle
            self.dimension = dimension
            self.last_point = None
            
        def setInteractive(self):
            self.interactive_planner = []

        def getInteractive(self):
            return self.interactive_planner

        def isValid(self,state):
            self.interactive_planner.append(state)            
            return self.clearence(state)

        def clearence(self,state):            
            if self.dimension == '2D':
                valid = True
                x = state[0]
                y = state[1]
                point_shp =  Point((x,y))
                region_shp = Polygon(self.region)
                if region_shp.contains(point_shp) or (x,y) == self.start or (x,y) == self.goal: #Inside region of interest
                    for polygon in self.obstacle:
                        polygon_shp = Polygon(polygon[0])
                        if polygon_shp.contains(point_shp): #Inside Obstacle
                            valid = False
                            return False
                        elif (x,y) == self.start or (x,y) == self.goal:
                            valid = True
                        else:
                            if self.last_point != None:
                                line = LineString([self.last_point, (x,y)])
                                if line.intersects(polygon_shp):
                                    valid = False
                                    return False
                    if valid == True:
                        self.last_point = (x,y)
                        return True
                    else:
                        return False
                else:
                        return False
            else:
                print('Wrong Dimension')


    class ClearanceObjective(ob.StateCostIntegralObjective):

        def __init__(self, si):
            super().__init__(si, True)
            self.si_ = si

        def motionCost(self,s1,s2):
            x1 = round(s1[1]*(z.shape[0]-1))
            y1 = round(s1[0]*(z.shape[1]-1))
            x2 = round(s2[1]*(z.shape[0]-1))
            y2 = round(s2[0]*(z.shape[1]-1))
            xx, yy, val = line_aa(x1, y1, x2, y2)
            cost = 0
            for idx in range(len(xx)-1):
                cost = cost + z[xx[idx+1]][yy[idx+1]]*0.01
            return ob.Cost(cost)


    # Keep these in alphabetical order and all lower case
    def allocateObjective(self, si, objectiveType):
        if objectiveType.lower() == "pathclearance":
            return self.getClearanceObjective(si)
        elif objectiveType.lower() == "pathlength":
            return self.getPathLengthObjective(si)
        elif objectiveType.lower() == "thresholdpathlength":
            return self.getThresholdPathLengthObj(si)
        elif objectiveType.lower() == "weightedlengthandclearancecombo":
            return self.getBalancedObjective1(si)
        else:
            ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

    # Keep these in alphabetical order and all lower case
    def allocatePlanner(self, si, plannerType):
        if plannerType.lower() == "bfmtstar":
            return og.BFMT(si)
        elif plannerType.lower() == "bitstar":
            return og.BITstar(si)
        elif plannerType.lower() == "fmtstar":
            return og.FMT(si)
        elif plannerType.lower() == "informedrrtstar":
            return og.InformedRRTstar(si)
        elif plannerType.lower() == "prmstar":
            return og.PRMstar(si)
        elif plannerType.lower() == "rrtstar":
            return og.RRTstar(si)
        elif plannerType.lower() == "sorrtstar":
            return og.SORRTstar(si)
        else:
            ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

    def getPathLengthObjective(self, si):
        return ob.PathLengthOptimizationObjective(si)

    def getThresholdPathLengthObj(self, si):
        obj = ob.PathLengthOptimizationObjective(si)
        obj.setCostThreshold(ob.Cost(2))
        return obj

    def getClearanceObjective(self, si):
        return self.ClearanceObjective(si)

    def getBalancedObjective1(self, si):
        lengthObj = ob.PathLengthOptimizationObjective(si)
        clearObj = self.ClearanceObjective(si)
        opt = ob.MultiOptimizationObjective(si)
        opt.addObjective(lengthObj, 1.0)
        opt.addObjective(clearObj, 1.0)
        return opt

    def getPathLengthObjWithCostToGo(self, si):
        obj = ob.PathLengthOptimizationObjective(si)
        obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
        return obj
    
    def OMPL_plan(self, runTime, plannerType, objectiveType):
        # Construct the robot state space in which we're planning. We're
        # planning in [0,1]x[0,1], a subset of R^2.
        if self.dimension == '2D':
            space = ob.RealVectorStateSpace(2)
            # Set the bounds of space to be in [0,1].
            bounds = ob.RealVectorBounds(2)
            x_bound = []
            y_bound = []
            for idx in range(len(self.region)):
                x_bound.append(self.region[idx][0])
                y_bound.append(self.region[idx][1])

            bounds.setLow(0,min(x_bound))
            bounds.setHigh(0,max(x_bound))
            bounds.setLow(1,min(y_bound))
            bounds.setHigh(1,max(y_bound))
            
            self.x_bound = (min(x_bound),max(x_bound))
            self.y_bound = (min(y_bound),max(y_bound))
            #test = bounds.getDifference()
            #test = bounds.check()

            space.setBounds(bounds)
            # Set our robot's starting state to be the bottom-left corner of
            # the environment, or (0,0).
            start = ob.State(space)
            start[0] = self.start[0]
            start[1] = self.start[1]
            #start[2] = 0.0  

            # Set our robot's goal state to be the top-right corner of the
            # environment, or (1,1).
            goal = ob.State(space)
            goal[0] = self.goal[0]
            goal[1] = self.goal[1]
            #goal[2] = 1.0
        else:
            pass

        # Construct a space information instance for this state space
        si = ob.SpaceInformation(space)

        # Set the object used to check which states in the space are valid
        
        validityChecker = self.ValidityChecker(si)
        validityChecker.setInteractive()
        validityChecker.obstacle(self.start, self.goal, self.region, self.obstacle, self.dimension)
        si.setStateValidityChecker(validityChecker)

        si.setup()

        # Create a problem instance
        pdef = ob.ProblemDefinition(si)

        # Set the start and goal states
        pdef.setStartAndGoalStates(start, goal)

        # Create the optimization objective specified by our command-line argument.
        # This helper function is simply a switch statement.
        pdef.setOptimizationObjective(self.allocateObjective(si, objectiveType))

        # Construct the optimal planner specified by our command line argument.
        # This helper function is simply a switch statement.
        optimizingPlanner = self.allocatePlanner(si, plannerType)

        # Set the problem instance for our planner to solve
        optimizingPlanner.setProblemDefinition(pdef)
        optimizingPlanner.setup()

        # attempt to solve the planning problem in the given runtime
        solved = optimizingPlanner.solve(runTime)

        self.PlannerStates.append((validityChecker.getInteractive(),plannerType))

        if solved:
            # Output the length of the path found
            try:
                print('{0} found solution of path length {1:.4f} with an optimization ' \
                    'objective value of {2:.4f}'.format( \
                    optimizingPlanner.getName(), \
                    pdef.getSolutionPath().length(), \
                    pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

                return self.decodeSolutionPath(pdef.getSolutionPath().printAsMatrix(),plannerType, pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value())
            except:
                print("No solution found.")
                pass
                
        else:
            print("No solution found.")
    
    def decodeSolutionPath(self, path, plannerType, pathCost):
        solution = []
        solution_lat = []
        solution_lon = []
        solution_planner = []
        path = path.replace('\n','')
        path = path.split(' ')
        path = path[:len(path)-1]
        for idx in range(int(len(path)/2)):
            solution_lat.append(float(path[2*idx]))
            solution_lon.append(float(path[2*idx+1]))
        #elif self.dimension == '3D':
        #    for idx in range(int(len(path)/3)):
        #        solution_lat.append(float(path[3*idx]))
        #        solution_lon.append(float(path[3*idx+1]))
        #        solution_alt.append(float(path[3*idx+2]))
        #        solution_planner.append(plannerType)

        #    df = pd.DataFrame({'latitude':solution_lat,'longitude':solution_lon,'altitude':solution_alt,'algorithm':solution_planner})
        #    self.solutionDataFrame = pd.concat([self.solutionDataFrame,df],ignore_index=True)


        #    self.solutionData.append((plannerType, df, pathCost))

        #    self.solution.append((solution_lat,solution_lon, solution_alt, plannerType, pathCost))
        #else:
        #    print('Error inside SolutionPath')
        self.solution.append((solution_lat,solution_lon,plannerType))
        return (solution_lat,solution_lon,plannerType)

    def plotSolutionPath(self,anima=False):
        if anima == False:
            if self.dimension == '2D':
                fig, ax = plt.subplots()
                #Obstacle
                for polygon in self.obstacle:
                    lat,lon = zip(*polygon[0])
                    lat = list(lat)
                    lon = list(lon)
                    lat.append(polygon[0][0][0])
                    lon.append(polygon[0][0][1])
                #    ax.plot(lon, lat, linestyle='-', color='red')
                    ax.fill(lon, lat, facecolor='gray', edgecolor='black')
                
                #Region
                lat,lon = zip(*self.region)
                lat = list(lat)
                lon = list(lon)
                lat.append(self.region[0][0])
                lon.append(self.region[0][1])
                ax.plot(lon, lat, linestyle='-.', color='green', label='Region of Interest')

                #Solution
                for solution in self.solution:
                    ax.plot(solution[1], solution[0], label=solution[2])
                ax.set(xlabel='Latitude', ylabel='Longitude', title='Solution Path')
                ax.legend()
                #ax.grid()
                #ax.autoscale()
                plt.show()

            elif self.dimension == '3D':
                
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
 
            else:
                print('Error inside plotSolutionPath')
            
        else:
            if self.dimension == '2D':
                fig, ax = plt.subplots()
                #fig, ax = plt.subplots()
                #Obstacle
                for polygon in self.obstacle:
                    x,y = zip(*polygon[0])
                    line, = ax.plot(x, y, 'r-')
                def init():
                    ax.set_xlim(self.x_bound[0]*1.1, self.x_bound[1]*1.1)
                    ax.set_ylim(self.y_bound[0]*1.1, self.y_bound[1]*1.1)
                    return ln,
                cont = 0
                def update(frame):
                    xdata.append(frame(0)[cont])
                    ydata.append(frame(1)[cont])
                    ln.set_data(xdata, ydata)
                    cont = cont + 1
                    return ln,
                #Solution
                for solution in self.solution:
                    xdata, ydata = [], []
                    ln, = plt.plot([], [], label=solution[2])
                    ani = FuncAnimation(fig, update, frames=(solution[0], solution[1]),
                                    init_func=init, blit=True)
                    ax.set(xlabel='Latitude', ylabel='Longitude',
                    title='Solution Path')
                    ax.legend()
                    #ax.grid()
                    plt.show()
                    input("Enter something")
            elif self.dimension == '3D':
                pass
            else:
                print('Error inside plotSolutionPath')

    def Plot_MPL(self):
    
            #Obstacle MatplotLib
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for solution in self.solution:
                ax.plot(solution[1], solution[0], solution[2], label=solution[3]) 

            lat_list = []
            lon_list = []
            alt_list = []
            lat_dist = []
            lon_dist = []
            alt_dist = []

            for polygon in self.obstacle:
                _lat = []
                _lon = []
                for point in polygon[0]:
                    _lat.append(point[0])
                    _lon.append(point[1])
                    
                lat_list.append(min(_lat))
                lon_list.append(min(_lon))
                alt_list.append(polygon[1])
                lat_dist.append(max(_lat)-min(_lat))
                lon_dist.append(max(_lon)-min(_lon))
                alt_dist.append(polygon[2])
            
            ax.bar3d(lon_list, lat_list, alt_list, lon_dist, lat_dist, alt_dist, shade=True, color='gray',alpha=1, zsort='max')
            ax.set(xlabel='Latitude', ylabel='Longitude',zlabel='Altitude', title='Solution Path')
            ax.legend()
            ax.grid()
            #ax.autoscale()
            plt.show()

    def plotOptimal(self, mult):
        import random
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt

        #data = np.loadtxt('path.txt')
        #fig, ax = plt.subplots()
        #ax = fig.gca(projection='3d')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #Obstacle
        for polygon in self.obstacle:
            lat,lon = zip(*polygon[0])
            lat = list(lat)
            lon = list(lon)
            lat.append(polygon[0][0][0])
            lon.append(polygon[0][0][1])
        #    ax.plot(lon, lat, linestyle='-', color='red')
            ax.fill(lon, lat, facecolor='gray', edgecolor='black')
        
        #Region
        lat,lon = zip(*self.region)
        lat = list(lat)
        lon = list(lon)
        lat.append(self.region[0][0])
        lon.append(self.region[0][1])
        ax.plot(lon, lat, linestyle='-.', color='green', label='Region of Interest')

        #Solution
        for sol in self.solution:
            ax.plot(sol[1], sol[0], label=sol[2])
        ax.set(xlabel='Latitude', ylabel='Longitude', title='Solution Path')
        ax.legend()
        #ax.grid()
        #ax.autoscale()
        #plt.show()



    
        

        #ellipse = Ellipse((0.5, 0.5), 0.6, 0.6)
        #ax.add_artist(ellipse)
    
        #ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
        #      extent=[x.min(), x.max(), y.min(), y.max()],
        #      interpolation='nearest', origin='lower', aspect='auto', animated=True)
        #if i == 0:
        #    ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
        #      extent=[x.min(), x.max(), y.min(), y.max()],
        #      interpolation='nearest', origin='lower', aspect='auto')
        ax.pcolormesh(x, y, z*0.02, cmap='RdBu', shading='nearest', vmin=z_min, vmax=z_max)
        X, Y = np.meshgrid(x, y)
        #ax.plot(X.flat, Y.flat, '.', color='m')
        try:
            for sol in self.solution:
                ax.plot(sol[1],sol[0],label=sol[2])
        except:
            pass
        ax.set_title('image (nearest, aspect="auto")')
        #fig.colorbar(im, ax=ax)
        ax.set(xlabel='Latitude', ylabel='Longitude',
        title='Simple Plot')
        ax.legend()

        for sol in self.solution:
            x_main=[]
            y_main=[]
            x_points = []
            y_points = []
            v_value = []
            img = np.zeros((z.shape[0], z.shape[1]), dtype=np.uint8)
            try:
                for idx in range(len(sol[0])-1):
                    x1 = round(sol[1][idx]*(z.shape[0]-1))
                    y1 = round(sol[0][idx]*(z.shape[1]-1))
                    x2 = round(sol[1][idx+1]*(z.shape[0]-1))
                    y2 = round(sol[0][idx+1]*(z.shape[1]-1))
                    x_main.append(x1*mult)
                    y_main.append(y1*mult)
                    x_main.append(x2*mult)
                    y_main.append(y2*mult)
                    xx, yy, val = line_aa(x1, y1, x2, y2)
                    for i_ in range(len(xx)):
                        x_points.append(xx[i_]*mult)
                        y_points.append(yy[i_]*mult)
                        v_value.append(val[i_]*mult)
                    img[xx,yy] = val
            except:
                pass
                #for i_ in range(len(xx)):
                #    img[xx[i_], yy[i_]] = val[i_]*100
            #ax.pcolormesh(x_points, y_points, v_value, cmap='RdBu', shading='nearest', vmin=v_value, vmax=v_value)
            self.solutionSampled.append((x_main,y_main))
            ax.plot(x_points,y_points,'.')
            ax.plot(x_main,y_main)

        #plt.show()
    

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=2.0, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')
    parser.add_argument('-p', '--planner', default='InformedRRTstar', \
        choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar'], \
        help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')
    parser.add_argument('-o', '--objective', default='PathLength', \
        choices=['PathClearance', 'PathLength', 'ThresholdPathLength', \
        'WeightedLengthAndClearanceCombo'], \
        help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')
    parser.add_argument('-f', '--file', default='path.txt', \
        help='(Optional) Specify an output path for the found solution path.')
    parser.add_argument('-i', '--info', type=int, default=0, choices=[0, 1, 2], \
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
        ' Defaults to WARN.')

    # Parse the arguments
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (args.runtime,))

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")


    #polygon = [(-12.0, -47.98), (-12.0, -46.99),
    #           (-12.0, -46.99), (-12.6, -46.99),
    #           (-12.6, -46.99), (-12.6, -47.98),
    #           (-12.6, -47.98), (-12.0, -47.98)]
    #base = 20
    #topo = 100
    #obstacle = [(polygon, base, topo)]



    # Solve the planning problem
    polygon = [(-3,-2),
               (-3, 5), 
               ( 6, 5),
               ( 6,-2)]
    base = 2
    topo = 6
    
    polygon2 = [(8, 9),
                (8, 6),
                (6, 6),
                (6, 9)]
    type_obstacle = 'CB'  

    obstacle = [(polygon, base, topo, type_obstacle),(polygon2, base, topo,type_obstacle)]
    #obstacle = [([(-12.200000000000001, -47.5), (-12.1, -47.5), (-12.1, -47.599999999999994), (-12.200000000000001, -47.599999999999994)], 0.2, 0.6), ([(-12.3, -47.5), (-12.2, -47.5), (-12.2, -47.599999999999994), (-12.3, -47.599999999999994)], 0.2, 0.6), ([(-12.3, -47.400000000000006), (-12.2, -47.400000000000006), (-12.2, -47.5), (-12.3, -47.5)], 0.2, 0.6), ([(-12.3, -47.1), (-12.2, -47.1), (-12.2, -47.199999999999996), (-12.3, -47.199999999999996)], 0.2, 0.6), ([(-12.4, -47.6), (-12.299999999999999, -47.6), (-12.299999999999999, -47.699999999999996), (-12.4, -47.699999999999996)], 0.2, 0.6), ([(-12.4, -47.5), (-12.299999999999999, -47.5), (-12.299999999999999, -47.599999999999994), (-12.4, -47.599999999999994)], 0.2, 0.6), ([(-12.4, -47.400000000000006), (-12.299999999999999, -47.400000000000006), (-12.299999999999999, -47.5), (-12.4, -47.5)], 0.2, 0.6), ([(-12.5, -47.7), (-12.399999999999999, -47.7), (-12.399999999999999, -47.8), (-12.5, -47.8)], 0.2, 0.6), ([(-12.5, -47.6), (-12.399999999999999, -47.6), (-12.399999999999999, -47.699999999999996), (-12.5, -47.699999999999996)], 0.2, 0.6), ([(-12.5, -47.5), (-12.399999999999999, -47.5), (-12.399999999999999, -47.599999999999994), (-12.5, -47.599999999999994)], 0.2, 0.6), ([(-12.5, -47.400000000000006), (-12.399999999999999, -47.400000000000006), (-12.399999999999999, -47.5), (-12.5, -47.5)], 0.2, 0.6), ([(-12.5, -47.300000000000004), (-12.399999999999999, -47.300000000000004), (-12.399999999999999, -47.4), (-12.5, -47.4)], 0.2, 0.6), ([(-12.600000000000001, -47.5), (-12.5, -47.5), (-12.5, -47.599999999999994), (-12.600000000000001, -47.599999999999994)], 0.2, 0.6), ([(-12.600000000000001, -47.400000000000006), (-12.5, -47.400000000000006), (-12.5, -47.5), (-12.600000000000001, -47.5)], 0.2, 0.6), ([(-12.600000000000001, -47.300000000000004), (-12.5, -47.300000000000004), (-12.5, -47.4), (-12.600000000000001, -47.4)], 0.2, 0.6)]

    start =(0.1,0.1,1) 
    goal = (0.9,0.9,8)
    region = [( 0, 0),
              ( 1, 0),
              ( 1, 1),
              ( 0, 1)]
    #start =(-12.62, -47.86, 0.2) 
    #goal = (-12.21, -47.28, 0.7)
    #region = [(-12.0, -47.98), 
    #        (-12.0, -46.99), 
    #        (-12.67, -46.99), 
    #        (-12.67, -47.98)]


    dimension = '2D'
    planner = 'RRTstar'


    polygon = [(0.45, 0.45), 
               (0.45, 0.55),
               (0.55, 0.55), 
               (0.55, 0.45)]
    base = 0.2
    topo = 0.6
    obstacle = [(polygon, base, topo)]



    # make these smaller to increase the resolution
    dx, dy = 0.015, 0.005

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[0:1+dy:dy, 0:1+dx:dx]

    rand = random.random()
    z = (rand + x/rand + x**5 + y**3) * np.exp(-x**2 + y**2)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = -abs(z).max(), abs(z).max()


    nrows = 10
    ncols = 10
    x = np.arange(ncols+1)*0.1
    y = np.arange(nrows+1)*0.1

    z =[[1,1,1,1,1,0,0,0,0,0,0],\
        [1,1,1,1,1,0,0,0,0,0,0],\
        [1,1,1,1,1,0,0,0,0,0,0],\
        [1,1,1,1,1,0,0,0,0,0,0],\
        [1,1,1,1,1,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,0,0,0,0,0],\
        [0,0,0,0,0,0,1,1,1,1,1],\
        [0,0,0,0,0,0,1,1,1,1,1],\
        [0,0,0,0,0,0,1,1,1,1,1],\
        [0,0,0,0,0,0,1,1,1,1,1]]

    z =[[10,10,10,10,10,10,10,10,10,10,10],\
        [10,20,20,20,20,20,20,20,20,20,10],\
        [10,20,50,50,50,50,50,50,50,20,10],\
        [10,20,50,100,100,100,100,100,50,20,10],\
        [10,20,50,100,200,200,200,100,50,20,10],\
        [10,20,50,100,200,300,200,100,50,20,10],\
        [10,20,50,100,200,200,200,100,50,20,10],\
        [10,20,50,100,100,100,100,100,50,20,10],\
        [10,20,50,50,50,50,50,50,50,20,10],\
        [10,20,20,20,20,20,20,20,20,20,10],\
        [10,10,10,10,10,10,10,10,10,10,10]]

    #z = np.asarray(z,dtype=np.double) 
    from skimage.draw import ellipse
    from skimage.draw import disk
    nrows = 10
    ncols = 10
    delta_d = 1/nrows
    x = np.arange(ncols+1)*0.1
    y = np.arange(nrows+1)*0.1
    z = np.zeros((nrows+1, ncols+1), dtype=np.uint8) + 1
    xx,yy = disk((5,5),5)
    z[xx,yy] = 10

    xx,yy = disk((5,5),2.50)
    z[xx,yy] = 20

    xx,yy = disk((5,5),1.25)
    z[xx,yy] = 50


    xx, yy = ellipse(5, 6, 1, 2, rotation=np.deg2rad(30))
    z[xx,yy] = 100

    xx, yy = ellipse(2, 4, 0.50, 2.50, rotation=np.deg2rad(10))
    z[xx,yy] = 100

    z = np.asarray(z,dtype=np.double) 
    print(z)

    run = True
    path_x = []
    path_y = []
    plans = []
    while (run == True):
        try:
            if len(plans) == 0:
                plans.append(OptimalPlanning(start, goal, region, obstacle, planner, dimension))
                result = plans[0].plan(2, 'RRTstar', 'WeightedLengthAndClearanceCombo')
                plans[0].plotOptimal(delta_d)
            else:
                #Linear algegra to return the next point in a line
                p1 = Vector2(plans[-1].solutionSampled[0][1][0], plans[-1].solutionSampled[0][0][0])
                p2 = Vector2(plans[-1].solutionSampled[0][1][1], plans[-1].solutionSampled[0][0][1])
                vector = p2-p1
                vector = vector.normalize()
                next_point = p1 + vector*delta_d

                plans.append(OptimalPlanning((next_point.x,next_point.y), goal, region, obstacle, planner, dimension))
                result = plans[-1].plan(2, 'RRTstar', 'WeightedLengthAndClearanceCombo')
                plans[-1].plotOptimal(delta_d)
                if (plans[-1].solutionSampled[0][1][0], plans[-1].solutionSampled[0][0][0]) == (goal[0],goal[1]):
                    run = False
            path_x.append(plans[-1].solutionSampled[0][1][0])
            path_y.append(plans[-1].solutionSampled[0][0][0])
        except:
            plans.pop()

    from matplotlib import pyplot as plt 
    import numpy as np 
    from matplotlib.animation import FuncAnimation  
    
    # initializing a figure in  
    # which the graph will be plottpath_ed 
    fig = plt.figure()  
    
    # marking the x-axis and y-axis 
    axis = plt.axes(xlim =(0, 1),  
                    ylim =(0, 1))  
    
    # initializing a line variable 
    line, = axis.plot([], [],'.', lw = 3)  
    axis.pcolormesh(x, y, z*0.02, cmap='RdBu', shading='nearest', vmin=z_min, vmax=z_max)
    # data which the line will  
    # contain (x, y) 
    def init():  
        line.set_data([], []) 
        return line, 

    #x = np.linspace(0, 4, 1000)
    #y = np.sin(2 * np.pi * (x - 0.01))
    def animate(i): 
    
        # plots a sine graph 
         
        line.set_data(path_y[:i], path_x[:i]) 
        
        return line, 
    
    anim = FuncAnimation(fig, animate, init_func = init, 
                        frames = 20, interval = 200, blit = True) 
    
    plt.show()


    #plan1 = OptimalPlanning((plan.solutionSampled[0][1][1], plan.solutionSampled[0][0][1]), goal, region, obstacle, planner, dimension)
    #result = plan1.plan(2, 'RRTstar', 'WeightedLengthAndClearanceCombo')
    #plan1.plotOptimal(0.1)
    

    #for planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar']:
    #for planner in ['RRTstar', 'InformedRRTstar']:
    #    result = plan.plan(2, planner, 'WeightedLengthAndClearanceCombo')
    #print(plan.solution)WeightedLengthAndClearanceCombo   PathClearance  PathLength
    #plan.plotSolutionPath(anima=False)
    #plan.plotOptimal(0.1)

    


