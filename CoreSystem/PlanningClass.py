#!/usr/bin/env python
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
# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as f

import pandas as pd
from math import sqrt
import argparse
from shapely.geometry import Point,LineString
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
mpl.use('tkAgg')
#mpl.rcParams['agg.path.chunksize'] = 10000000
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
from matplotlib.collections import PolyCollection


import plotly.graph_objects as go
import plotly.express as px      

class PathPlanning():
    def __init__(self, start, goal, region, obstacle, planner, dimension):
        self.start = start
        self.goal = goal
        self.region = region
        self.obstacle = obstacle
        self.planner = planner
        self.dimension = dimension
        self.solution = []
        self.PlannerStates = []
        self.solutionDataFrame = pd.DataFrame({'latitude':[],'longitude':[],'altitude':[],'algorithm':[]})
        
    

    def plan(self,runTime, plannerType, objectiveType):
        if self.planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar']:
            result = self.OMPL_plan(runTime, plannerType, objectiveType)
            return result
        elif self.planner == 'PSO':
            result = self.PSO_plan()
            return result
        else:
            return False
            pass

    def PSO_plan():
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

            elif self.dimension == '3D':
                valid = True
                x = state[0]
                y = state[1]
                z = state[2]
                if self.last_point != None and (x,y) != (self.start[0],self.start[1]) and (x,y) != (self.start[0],self.start[1]):
                    for polygon in self.obstacle:
                        polygon_shp = Polygon(polygon[0])
                        line = LineString([self.last_point, (x,y)])
                        if line.intersects(polygon_shp):
                            if z > polygon[1] or z < polygon[2]: 
                                valid = False
                                return False
                    if valid == True:
                        self.last_point = (x,y)
                        return True
                else:
                    self.last_point = (x,y)
                    return True
            else:
                print('Wrong Dimension')
 

    class ClearanceObjective(ob.StateCostIntegralObjective):
        def __init__(self, si):
            super(ClearanceObjective, self).__init__(si, True)
            self.si_ = si

        # Our requirement is to maximize path clearance from obstacles,
        # but we want to represent the objective as a path cost
        # minimization. Therefore, we set each state's cost to be the
        # reciprocal of its clearance, so that as state clearance
        # increases, the state cost decreases.
        def stateCost(self, s):
            return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
                                sys.float_info.min))


    def getPathLengthObjective(self,si):
        return ob.PathLengthOptimizationObjective(si)

    def getThresholdPathLengthObj(self,si):
        obj = ob.PathLengthOptimizationObjective(si)
        obj.setCostThreshold(ob.Cost(1.51))
        return obj

    def getClearanceObjective(self,si):
        return ClearanceObjective(si)

    def getBalancedObjective1(self,si):
        lengthObj = ob.PathLengthOptimizationObjective(si)
        clearObj = ClearanceObjective(si)

        opt = ob.MultiOptimizationObjective(si)
        opt.addObjective(lengthObj, 5.0)
        opt.addObjective(clearObj, 1.0)
        return opt

    def getPathLengthObjWithCostToGo(self,si):
        obj = ob.PathLengthOptimizationObjective(si)
        obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
        return obj

    # Keep these in alphabetical order and all lower case
    def allocatePlanner(self,si, plannerType):
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

    # Keep these in alphabetical order and all lower case
    def allocateObjective(self,si, objectiveType):
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
        elif self.dimension == '3D':
            space = ob.RealVectorStateSpace(3)
            bounds = ob.RealVectorBounds(3)
            x_bound = []
            y_bound = []
            z_bound = [0,10]
            for idx in range(len(self.region)):
                x_bound.append(self.region[idx][0])
                y_bound.append(self.region[idx][1])
                #z_bound.append(self.region[idx][2])
        
            bounds.setLow(0,min(x_bound))
            bounds.setHigh(0,max(x_bound))
            bounds.setLow(1,min(y_bound))
            bounds.setHigh(1,max(y_bound))
            bounds.setLow(2,min(z_bound))
            bounds.setHigh(2,max(z_bound))
            
            self.x_bound = (min(x_bound),max(x_bound))
            self.y_bound = (min(y_bound),max(y_bound))
            self.z_bound = (min(z_bound),max(z_bound))
            # Set the bounds of space to be in [0,1].
            space.setBounds(bounds)
            # Set our robot's starting state to be the bottom-left corner of
            # the environment, or (0,0).
            start = ob.State(space)
            start[0] = self.start[0]
            start[1] = self.start[1]
            start[2] = self.start[2]

            # Set our robot's goal state to be the top-right corner of the
            # environment, or (1,1).
            goal = ob.State(space)
            goal[0] = self.goal[0]
            goal[1] = self.goal[1]
            goal[2] = self.goal[2]
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

        # Set our robot's starting state to be the bottom-left corner of
        # the environment, or (0,0).
        #start = ob.State(space)
        #start[0] = self.start[0]
        #start[1] = self.start[1]
        #start[2] = 0.0  

        # Set our robot's goal state to be the top-right corner of the
        # environment, or (1,1).
        #goal = ob.State(space)
        #goal[0] = self.goal[0]
        #goal[1] = self.goal[1]
        #goal[2] = 1.0

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

                return self.decodeSolutionPath(pdef.getSolutionPath().printAsMatrix(),plannerType)
            except:
                print("No solution found.")
                pass
                
            # If a filename was specified, output the path as a matrix to
            # that file for visualization
            #path = og.PathGeometric(pdef.getSpaceInformation())
            #test = path.printAsMatrix()
            #simplifier = og.PathSimplifier(pdef.getSpaceInformation(),pdef.getGoal(),pdef.getOptimizationObjective())
            #path_simp = simplifier.simplifyMax(path)
            #test = path.printAsMatrix()
            #test = pdef.getSolutionPath().printAsMatrix()

            
            
            #if fname:
            #    with open(fname, 'w') as outFile:
            #        outFile.write(pdef.getSolutionPath().printAsMatrix())
            
        else:
            print("No solution found.")

    def decodeSolutionPath(self, path, plannerType):
        solution_lat = []
        solution_lon = []
        solution_alt = []
        solution_planner = []
        path = path.replace('\n','')
        path = path.split(' ')
        path = path[:len(path)-1]
        if self.dimension == '2D':
            for idx in range(int(len(path)/2)):
                solution_lat.append(float(path[2*idx]))
                solution_lon.append(float(path[2*idx+1]))
            self.solution.append((solution_lat,solution_lon,plannerType))
        elif self.dimension == '3D':
            for idx in range(int(len(path)/3)):
                solution_lat.append(float(path[3*idx]))
                solution_lon.append(float(path[3*idx+1]))
                solution_alt.append(float(path[3*idx+2]))
                solution_planner.append(plannerType)

            df = pd.DataFrame({'latitude':solution_lat,'longitude':solution_lon,'altitude':solution_alt,'algorithm':solution_planner})
            self.solutionDataFrame = pd.concat([self.solutionDataFrame,df],ignore_index=True)

            self.solution.append((solution_lat,solution_lon, solution_alt,plannerType))
            
        else:
            print('Error inside SolutionPath')
    
        return self.solution


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
                fig = px.line_3d(self.solutionDataFrame, x="latitude", y="longitude", z="altitude",color='algorithm')
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
                    name='y',
                    showscale=True
                    ))
                
                fig.update_scenes(xaxis_autorange="reversed")
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

    def Plot_MPL():
    
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

    def plotPlannerStates(self):
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if self.dimension == '2D':
            fig, ax = plt.subplots()
            #Obstacle
            for polygon in self.obstacle:
                x,y = zip(*polygon[0])
                line, = ax.plot(x, y, 'r-')
                
            #Solution
            for solution in self.PlannerStates:
                ax.plot(solution[0], solution[1], label=solution[2])

            ax.set(xlabel='Latitude', ylabel='Longitude',
                title='Solution Path')
            ax.legend()
            #ax.grid()
            plt.show()

        elif self.dimension == '3D':
            pass
        else:
            print('Error inside plotSolutionPath')
    



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


    obstacle = [(polygon, base, topo),(polygon2, base, topo)]
    #obstacle = [([(-12.200000000000001, -47.5), (-12.1, -47.5), (-12.1, -47.599999999999994), (-12.200000000000001, -47.599999999999994)], 0.2, 0.6), ([(-12.3, -47.5), (-12.2, -47.5), (-12.2, -47.599999999999994), (-12.3, -47.599999999999994)], 0.2, 0.6), ([(-12.3, -47.400000000000006), (-12.2, -47.400000000000006), (-12.2, -47.5), (-12.3, -47.5)], 0.2, 0.6), ([(-12.3, -47.1), (-12.2, -47.1), (-12.2, -47.199999999999996), (-12.3, -47.199999999999996)], 0.2, 0.6), ([(-12.4, -47.6), (-12.299999999999999, -47.6), (-12.299999999999999, -47.699999999999996), (-12.4, -47.699999999999996)], 0.2, 0.6), ([(-12.4, -47.5), (-12.299999999999999, -47.5), (-12.299999999999999, -47.599999999999994), (-12.4, -47.599999999999994)], 0.2, 0.6), ([(-12.4, -47.400000000000006), (-12.299999999999999, -47.400000000000006), (-12.299999999999999, -47.5), (-12.4, -47.5)], 0.2, 0.6), ([(-12.5, -47.7), (-12.399999999999999, -47.7), (-12.399999999999999, -47.8), (-12.5, -47.8)], 0.2, 0.6), ([(-12.5, -47.6), (-12.399999999999999, -47.6), (-12.399999999999999, -47.699999999999996), (-12.5, -47.699999999999996)], 0.2, 0.6), ([(-12.5, -47.5), (-12.399999999999999, -47.5), (-12.399999999999999, -47.599999999999994), (-12.5, -47.599999999999994)], 0.2, 0.6), ([(-12.5, -47.400000000000006), (-12.399999999999999, -47.400000000000006), (-12.399999999999999, -47.5), (-12.5, -47.5)], 0.2, 0.6), ([(-12.5, -47.300000000000004), (-12.399999999999999, -47.300000000000004), (-12.399999999999999, -47.4), (-12.5, -47.4)], 0.2, 0.6), ([(-12.600000000000001, -47.5), (-12.5, -47.5), (-12.5, -47.599999999999994), (-12.600000000000001, -47.599999999999994)], 0.2, 0.6), ([(-12.600000000000001, -47.400000000000006), (-12.5, -47.400000000000006), (-12.5, -47.5), (-12.600000000000001, -47.5)], 0.2, 0.6), ([(-12.600000000000001, -47.300000000000004), (-12.5, -47.300000000000004), (-12.5, -47.4), (-12.600000000000001, -47.4)], 0.2, 0.6)]

    start =(-10,-10,1) 
    goal = (10,10,8)
    region = [( 10, 10),
              ( 30,-10),
              ( 20,-20),
              (-10,-20),
              (-30, 0)]
    #start =(-12.62, -47.86, 0.2) 
    #goal = (-12.21, -47.28, 0.7)
    #region = [(-12.0, -47.98), 
    #        (-12.0, -46.99), 
    #        (-12.67, -46.99), 
    #        (-12.67, -47.98)]


    dimension = '3D'
    planner = 'RRTstar'

    plan = PathPlanning(start, goal, region, obstacle, planner, dimension)
    #result = plan.plan(2, planner, 'PathLength')

    for planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar']:
        result = plan.plan(2, planner, 'PathLength')
    print(plan.solution)
    plan.plotSolutionPath(anima=False)

    




                    #_ = []
                    #y_ = []
                    #for point in polygon[0]:
                    #    x_.append(point[0])
                    #    y_.append(point[1])
                    #xmin = min(x_)
                    #xmax = max(x_)
                    #ymin = min(y_)
                    #ymax = max(y_)

                    #z_z = [polygon[1],polygon[2]]
                    #poly = PolyCollection([polygon[0],polygon[0]], facecolors=['r'], alpha=.5)
                    #ax.add_collection3d(poly, zs=z_z, zdir='z')
                    
                    #z_y = [ymin,ymax]
                    #poly_ = [(xmin,polygon[1]),(xmin,polygon[2]),
                    #        (xmax,polygon[2]),(xmax,polygon[1])]
                    #poly = PolyCollection([poly_,poly_], facecolors=['r'], alpha=.5)
                    #ax.add_collection3d(poly, zs=z_y, zdir='y')

                    #z_x = [xmin,xmax]
                    #poly_ = [(ymin,polygon[1]),(ymin,polygon[2]),
                    #        (ymax,polygon[2]),(ymax,polygon[1])]
                    #poly = PolyCollection([poly_,poly_], facecolors=['r'], alpha=.5)
                    #ax.add_collection3d(poly, zs=z_x, zdir='x')
                    
                    #z_x = [polygon[0][0][0],polygon[0][3][0]]
                    #poly_ = [(polygon[0][0][1],polygon[1]),(polygon[0][1][1],polygon[1]),
                    #        (polygon[0][2][1],polygon[2]),(polygon[0][3][1],polygon[2])]
                    #poly = PolyCollection([poly_], facecolors=['r'], alpha=.6)
                    #ax.add_collection3d(poly, zs=z_x, zdir='x')