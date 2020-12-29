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
import random
from euclid import *
from MapGenClass import *
from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot
from skimage.draw import *

    
class OptimalPlanning():
    def __init__(self, start, goal, region, obstacle, planner, dimension, costmap):
        self.start = start
        self.goal = goal
        self.region = region
        self.obstacle = obstacle
        self.planner = planner
        self.dimension = dimension
        self.costmap = costmap
        self.solution = []
        self.solution_dict = {}
        self.solutionSampled = []
        self.PlannerStates = []

    def plan(self,runTime, plannerType, objectiveType,mult):
        if self.planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar']:
            result = self.OMPL_plan(runTime, plannerType, objectiveType,mult)
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

        def __init__(self, si, costmap):
            super().__init__(si, True)
            self.si_ = si
            self.costmap = costmap

        def motionCost(self,s1,s2):
            x1 = round(s1[1]*(self.costmap.shape[0]-1))
            y1 = round(s1[0]*(self.costmap.shape[1]-1))
            x2 = round(s2[1]*(self.costmap.shape[0]-1))
            y2 = round(s2[0]*(self.costmap.shape[1]-1))
            xx, yy, val = line_aa(x1, y1, x2, y2)
            cost = 0
            for idx in range(len(xx)-1):
                cost = cost + self.costmap[xx[idx+1]][yy[idx+1]]*0.1
            #cost = (cost/(len(xx)))
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
        elif plannerType.lower() == "cforest":
            return og.CForest(si)
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
        clearObj = self.ClearanceObjective(si,self.costmap)
        opt = ob.MultiOptimizationObjective(si)
        opt.addObjective(lengthObj, 1.0)
        opt.addObjective(clearObj, 1.0)
        return opt

    def getPathLengthObjWithCostToGo(self, si):
        obj = ob.PathLengthOptimizationObjective(si)
        obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
        return obj
    
    def OMPL_plan(self, runTime, plannerType, objectiveType,mult):
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

                return self.decodeSolutionPath(pdef.getSolutionPath().printAsMatrix(),plannerType, pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value(),mult)
            except:
                print("No solution found.")
                pass
                
        else:
            print("No solution found.")
    
    def decodeSolutionPath(self, path, plannerType, pathCost,mult):
        solution = []
        sampled_lat = []
        sampled_lon = []
        solution_lat = []
        solution_lon = []
        solution_planner = []
        path = path.replace('\n','')
        path = path.split(' ')
        path = path[:len(path)-1]
        for idx in range(int(len(path)/2)):
            solution_lat.append(float(path[2*idx]))
            solution_lon.append(float(path[2*idx+1]))
        self.solution.append((solution_lat,solution_lon,plannerType,pathCost))

        for idx in range(len(solution_lat)):
            sampled_lat.append(round(solution_lat[idx]*(self.costmap.shape[0]-1))*mult)
            sampled_lon.append(round(solution_lon[idx]*(self.costmap.shape[1]-1))*mult)
        self.solutionSampled.append((sampled_lat,sampled_lon))

        dict_1 = {"solution":{"lat":solution_lat,"lon":solution_lon,"type":plannerType, "pathCost":pathCost}}
        dict_2 = {"solutionSampled":{"lat":sampled_lat,"lon":sampled_lon,"type":plannerType, "pathCost":pathCost}}
        
        self.solution_dict.update(dict_1)
        self.solution_dict.update(dict_2)
            
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


    def plotOptimal(self, mult, axis):
        import random
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt

        #data = np.loadtxt('path.txt')
        #fig, ax = plt.subplots()
        #ax = fig.gca(projection='3d')
        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        aux = []
        #Obstacle
        for polygon in self.obstacle:
            lat,lon = zip(*polygon[0])
            lat = list(lat)
            lon = list(lon)
            lat.append(polygon[0][0][0])
            lon.append(polygon[0][0][1])
        #    ax.plot(lon, lat, linestyle='-', color='red')
            aux.append(axis.fill(lon, lat, facecolor='gray', edgecolor='black'))
            aux[-1]=aux[-1][0]
        
        #Region
        lat,lon = zip(*self.region)
        lat = list(lat)
        lon = list(lon)
        lat.append(self.region[0][0])
        lon.append(self.region[0][1])
        aux.append(axis.plot(lon, lat, linestyle='-.', color='green', label='Region of Interest'))
        aux[-1]=aux[-1][0]
        #Solution
        for sol in self.solution:
            aux.append(axis.plot(sol[0], sol[1], label=sol[2]))
            aux[-1]=aux[-1][0]
        aux.append(axis.plot(self.solutionSampled[0][0], self.solutionSampled[0][1], label='SolutionSampled'))
        aux[-1]=aux[-1][0]   
        axis.set(xlabel='Latitude', ylabel='Longitude', title='Solution Path')
        axis.legend()
        #ax.grid()
        #ax.autoscale()
        #plt.show()
        return aux



    
        

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

    #start =(0.1,0.1,1) 
    #goal = (0.9,0.9,8)
    start =(0.9,0.1,1) 
    goal = (0.1,0.9,1)
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


    ims = []
    plans = []
    path_x = []
    path_y = []
    run = True
    time = 8
    nrows = 80
    ncols = 80
    delta_d = 1/nrows
    fig = plt.figure()
    axis = plt.axes(xlim =(-0.2, 1.2),ylim =(-0.2, 1.2))
    mapgen = MapGen(nrows, ncols,time)
    mapgen.create()
    
    t = 0
    last_t = 0
    tried = 0
    max_try = 4
    pnt = 0
    while (run == True):
        #ims = []
        #fig = plt.figure()
        #axis = plt.axes(xlim =(-0.2, 1.2),ylim =(-0.2, 1.2))
        #try:
        plan_aux = []
        cost_aut = []
        test =[]
        if len(plans) == 0:
            for idx, alg in enumerate(['RRTstar']):
                plan_aux.append(OptimalPlanning(start, goal, region, [], planner, dimension, mapgen.z_time[round(t)]))
                result = plan_aux[idx].plan(15, alg, 'WeightedLengthAndClearanceCombo',delta_d)
                test.append(plan_aux[idx].plotOptimal(delta_d,axis))
                if plan_aux[idx].solution != []:
                    cost_aut.append(plan_aux[idx].solution[0][3])
                else:
                    cost_aut.append(np.inf)
            lower_cost = cost_aut.index(min(cost_aut))
            plans.append(plan_aux[lower_cost])

            print('Add, Start Solution: ' +  str(last_t) + " : " + str(round(t)) + " : " + str(t))
            
            path_x.append(plans[-1].solution[0][0][0])
            path_y.append(plans[-1].solution[0][1][0])
            aux = mapgen.plot_map(round(t),axis) + test[0]
            aux.append(axis.plot(path_x,path_y,'.',lw=3,color='black'))
            aux[-1] = aux[-1][0]
            aux = tuple(aux)
            ims.append(aux)

            last_t = round(t)
            t = t + 0.1
            if t >= time-1:
                t = time-1
        

        else:
            if tried <= max_try:

                #Linear algegra to return the next point in a line
                p1 = Vector2(plans[-1].solutionSampled[0][0][0], plans[-1].solutionSampled[0][1][0])
                p2 = Vector2(plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                #p1 = Vector2(plans[-1].solution[0][1][0], plans[-1].solution[0][0][0])
                #p2 = Vector2(plans[-1].solution[0][1][1], plans[-1].solution[0][0][1])
                vector = p2-p1
                vector = vector.normalize()
                next_point = p1 + vector*delta_d
            else:
                p1 = Vector2(plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                p2 = Vector2(plans[-1].solutionSampled[0][0][2], plans[-1].solutionSampled[0][1][2])
                vector = p2-p1
                vector = vector.normalize()
                next_point = p1 + vector*delta_d
                #next_point = (plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                

            for idx, alg in enumerate(['RRTstar']):
                plan_aux.append(OptimalPlanning((next_point[0],next_point[1]), goal, region, [], planner, dimension, mapgen.z_time[round(t)]))
                result = plan_aux[idx].plan(3, alg, 'WeightedLengthAndClearanceCombo',delta_d)
                if plan_aux[idx].solution != []:
                    cost_aut.append(plan_aux[idx].solution[0][3])
                else:
                    cost_aut.append(np.inf)
            lower_cost = cost_aut.index(min(cost_aut))
            #Se existir solução
            if plan_aux[lower_cost].solution != []:
                #Se o ultimo costmap é do mesmo periodo que o atual
                if round(t) == last_t:
                    #Tentar encontrar um valor melhor que o ultimo
                    if plan_aux[lower_cost].solution[0][3] < plans[-1].solution[0][3] * 1.10 :
                        print('Add, Better Solution: ' +  str(last_t) + " : " + str(round(t)) + " : " + str(t) + " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                        test.append(plan_aux[lower_cost].plotOptimal(delta_d,axis))
                        plans.append(plan_aux[lower_cost])
                        path_x.append(plans[-1].solution[0][0][0])
                        path_y.append(plans[-1].solution[0][1][0])

                        aux = mapgen.plot_map(round(t),axis) + test[0]
                        aux.append(axis.plot(path_x,path_y,'.',lw=3,color='black'))
                        aux[-1] = aux[-1][0]
                        aux = tuple(aux)
                        ims.append(aux)
                        
                        last_t = round(t)
                        t = t + 0.1
                        if t >= time-1:
                            t = time-1
                        
                        tried = 0
                        pnt = 1

                    elif tried >= max_try: 
                        print('Add, tried Solution: ' +  str(last_t) + " : " + str(round(t)) + " : " + str(t) + " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                        test.append(plan_aux[lower_cost].plotOptimal(delta_d,axis))
                        plans.append(plan_aux[lower_cost])
                        path_x.append(plans[-1].solution[0][0][0])
                        path_y.append(plans[-1].solution[0][1][0])

                        aux = mapgen.plot_map(round(t),axis) + test[0]
                        aux.append(axis.plot(path_x,path_y,'.',lw=3,color='black'))
                        aux[-1] = aux[-1][0]
                        aux = tuple(aux)
                        ims.append(aux)
                        
                        last_t = round(t)
                        t = t + 0.1
                        if t >= time-1:
                            t = time-1
                        
                        tried = 0
                        pnt = 1
                    else:
                        print('Tried: '+ str(tried) + " - (" + str(next_point[0])+','+str(next_point[1])+")" )
                        if tried >= max_try :
                            pnt = pnt +1    
                        tried = tried + 1

                else:
                    print('Add, Other time: ' +  str(last_t) + " : " + str(round(t)) + " : " + str(t)+ " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                    test.append(plan_aux[lower_cost].plotOptimal(delta_d,axis))
                    plans.append(plan_aux[lower_cost])
                    path_x.append(plans[-1].solution[0][0][0])
                    path_y.append(plans[-1].solution[0][1][0])

                    aux = mapgen.plot_map(round(t),axis) + test[0]
                    aux.append(axis.plot(path_x,path_y,'.',lw=3,color='black'))
                    aux[-1] = aux[-1][0]
                    aux = tuple(aux)
                    ims.append(aux)
                    
                    last_t = round(t)
                    t = t + 0.1
                    if t >= time-1:
                        t = time-1
            
                
            else:
                print('Pop Solution - Time: ' +  str(round(t)) + " : " + str(t))
                continue
            
        

        #if (plans[-1].solutionSampled[0][1][0], plans[-1].solutionSampled[0][0][0]) == (plans[-2].solutionSampled[0][1][0], plans[-2].solutionSampled[0][0][0]):
        #    print('Pop Solution')
        #    plans.pop()

        
            
           # plans[-1]planspend(plans[-1].solutionSampled[0][0][0])
            #path_x.append(plans[-1].solution[0][1][0])
            #path_y.append(plans[-1].solution[0][0][0])

        if (plans[-1].solution[0][0][0], plans[-1].solution[0][1][0]) == (goal[0],goal[1]):
        #if (plans[-1].solution[0][1][0], plans[-1].solution[0][0][0]) == (goal[0],goal[1]):
            run = False
            im_ani = animation.ArtistAnimation(fig, ims, interval=3000/time)
            plt.show()
        else:
            im_ani = animation.ArtistAnimation(fig, ims, interval=3000/time)
            #plt.show()
        
        
        
        #except Ex:
        #    print(Ex)
        #    pass
            #plans.pop()

    from matplotlib import pyplot as plt 
    import numpy as np 
    from matplotlib.animation import FuncAnimation 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import pyplot
    
    im_ani = animation.ArtistAnimation(fig, ims, interval=10000/time, blit=True)
    #anim = FuncAnimation(fig, animate, init_func = init, 
    #                    frames =len(path_y) , interval = 200, blit = True) 
    
    plt.show()


    # initializing a figure in  
    # which the graph will be plottpath_ed 
    fig = plt.figure()  
    
    # marking the x-axis and y-axis 
    axis = plt.axes(xlim =(-0.2, 1.2),  
                    ylim =(-0.2, 1.2))  
    
    # initializing a line variable 
    #axis.pcolormesh(x, y, z*0.02, cmap='RdBu', shading='nearest', vmin=z_min, vmax=z_max)
    line, = axis.plot([], [],'.', lw = 3) 

    # data which the line will  
    # contain (x, y) 
    def init():  
        line.set_data([], []) 
        return line, 

    #x = np.linspace(0, 4, 1000)
    #y = np.sin(2 * np.pi * (x - 0.01))
    def animate(i): 
    
        # plots a sine graph 
         
        line.set_data(path_x[:i], path_y[:i])
        
        return line, 
    
    anim = FuncAnimation(fig, animate, init_func = init, 
                        frames =len(path_y) , interval = 200, blit = True) 
    
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

    


