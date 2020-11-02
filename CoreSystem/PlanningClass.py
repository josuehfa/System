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
from pyswarms.utils.functions import single_obj as fx
from math import sqrt
import argparse
from shapely.geometry import Point,LineString
from shapely.geometry.polygon import Polygon


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

    def plan(self,runTime, plannerType, objectiveType, fname):
        if self.planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar']:
            result = self.OMPL_plan(runTime, plannerType, objectiveType, fname)
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
        def obstacle(self,start, goal, obstacle,dimension):
            self.obstacle = obstacle
            self.dimension = dimension
            self.last_point = None
            self.start = start 
            self.goal = goal

        def setInteractive(self):
            self.interactive_planner = []

        def getInteractive(self):
            return self.interactive_planner

        def isValid(self,state):
            self.interactive_planner.append(state)            
            return self.clearence(state)

        def clearence(self,state):            
            if dimension == '2D':
                valid = True
                x = state[0]
                y = state[1]
                for polygon in self.obstacle:
                    polygon_shp = Polygon(polygon[0])
                    point_shp =  Point((x,y))
                    if polygon_shp.contains(point_shp):
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

            elif dimension == '3D':
                x = state[0]
                y = state[1]
                z = state[2]
                for polygon in self.obstacle:
                    polygon_shp = Polygon(polygon[0])
                    point_shp =  Point((x,y))
                    if polygon_shp.contains(point_shp):
                        if z > polygon[1] or z < polygon[2]: 
                            return False
                        else:
                            return True
                    else:
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

    def decodeSolutionPath(self, path, plannerType):
        solution_lat = []
        solution_lon = []
        solution_alt = []
        path = path.replace('\n','')
        path = path.split(' ')
        path = path[:len(path)-1]
        if self.dimension == '2D':
            for idx in range(int(len(path)/2)):
                solution_lat.append(float(path[2*idx]))
                solution_lon.append(float(path[2*idx+1]))
            self.solution.append((solution_lat,solution_lon,plannerType))
        elif self.dimension == '3D':
            for idx in range(len(path)/3):
                solution_lat.append(float(path[3*idx]))
                solution_lon.append(float(path[3*idx+1]))
                solution_alt.append(float(path[3*idx+2]))
                self.solution.append((solution_lat,solution_lon, solution_alt,plannerType))
        else:
            print('Error inside SolutionPath')
    
        return self.solution


    def plotSolutionPath(self,anima=False):
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as FuncAnimation

        if anima == False:
            if self.dimension == '2D':
                fig, ax = plt.subplots()
                #Obstacle
                for polygon in self.obstacle:
                    x,y = zip(*polygon[0])
                    line, = ax.plot(x, y, 'r-')
                #Solution
                for solution in self.solution:
                    ax.plot(solution[0], solution[1], label=solution[2])
                ax.set(xlabel='Latitude', ylabel='Longitude',
                    title='Solution Path')
                ax.legend()
                ax.grid()
                plt.show()
            elif self.dimension == '3D':
                pass
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
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
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
                    ax.grid()
                    plt.show()
                    input("Enter something")
            elif self.dimension == '3D':
                pass
            else:
                print('Error inside plotSolutionPath')




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
    


        #for polygon in self.obstacle:
        #    _x = []
        #    _y = []
        #    for point in polygon[0]:
        #        _x.append(point[0])
        #        _y.append(point[1])
            # setup the figure and axes
            #fig = plt.figure(figsize=(8, 3))
            #ax1 = fig.add_subplot(121, projection='3d')
            #ax2 = fig.add_subplot(122, projection='3d')

            # fake data
            #_x = np.arange(4)
            #_y = np.arange(5)
            #_xx, _yy = np.meshgrid(_x, _y)
            #x, y = _xx.ravel(), _yy.ravel()

            #top = x + y
            #bottom = np.zeros_like(top)
        #    width = depth = 1

            #ax.bar3d(_x, _y, polygon[1], width, depth, polygon[2], shade=True)
            #ax1.set_title('Shaded')

            #ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
            #ax2.set_title('Not Shaded')       
        



    def OMPL_plan(self,runTime, plannerType, objectiveType, fname):
        # Construct the robot state space in which we're planning. We're
        # planning in [0,1]x[0,1], a subset of R^2.
        space = ob.RealVectorStateSpace(2)

        # Set the bounds of space to be in [0,1].
        space.setBounds(-1.0, 1.0)

        # Construct a space information instance for this state space
        si = ob.SpaceInformation(space)

        # Set the object used to check which states in the space are valid
        
        validityChecker = self.ValidityChecker(si)
        validityChecker.setInteractive()
        validityChecker.obstacle(self.start,self.goal,self.obstacle,self.dimension)
        si.setStateValidityChecker(validityChecker)

        si.setup()

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
            print('{0} found solution of path length {1:.4f} with an optimization ' \
                'objective value of {2:.4f}'.format( \
                optimizingPlanner.getName(), \
                pdef.getSolutionPath().length(), \
                pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

            # If a filename was specified, output the path as a matrix to
            # that file for visualization
            return self.decodeSolutionPath(pdef.getSolutionPath().printAsMatrix(),plannerType)
            
            #if fname:
            #    with open(fname, 'w') as outFile:
            #        outFile.write(pdef.getSolutionPath().printAsMatrix())
        else:
            print("No solution found.")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=10.0, help=\
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
    polygon = [(-0.3, -0.2), (-0.3,0.5),
        (-0.3, 0.5), (0.6,0.5),
        (0.6, 0.5), (0.6,-0.2),
        (0.6, -0.2), (-0.3,-0.2)]
    base = 0.2
    topo = 0.6
    
    polygon2 = [(0.8, 0.9), (0.8,0.6),
        (0.8, 0.6), (0.6,0.6),
        (0.6, 0.6), (0.6,0.9),
        (0.6, 0.9), (0.8,0.9)]


    obstacle = [(polygon, base, topo),(polygon2, base, topo)]

    start =(-1,-1) 
    goal = (1,1)
    region = [(1, 1), (-1,1),
        (-1, 1), (-1,-1),
        (-1, -1), (1,-1),
        (1, -1), (1,1)]
    
    dimension = '2D'
    planner = 'RRTstar'

    plan = PathPlanning(start, goal, region, obstacle, planner, dimension)

    for planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar']:
        result = plan.plan(args.runtime, planner, args.objective, args.file)

    plan.plotSolutionPath(anima=False)

    
 
