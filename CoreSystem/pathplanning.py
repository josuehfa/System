#!/usr/bin/env python
# Author: Luis G. Torres, Mark Moll

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
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Ellipse
import numpy as np
import random
from skimage.draw import line_aa

polygon = [(0.3, 0.2), (0.3,0.5),
           (0.3, 0.5), (0.6,0.5),
           (0.6, 0.5), (0.6,0.2),
           (0.6, 0.2), (0.3,0.2)]
base = 0.2
topo = 0.6
obstacle = [(polygon, base, topo)]



# make these smaller to increase the resolution
dx, dy = 0.005, 0.005

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[0:1+dy:dy, 0:1+dx:dx]

rand = random.random()
z = abs((rand - x/rand - x**5 + y**3) * np.exp(-x**2 - y**2))
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
nrows = 100
ncols = 100
mult = 0.01
x = np.arange(ncols+1)*mult
y = np.arange(nrows+1)*mult
z = np.zeros((nrows+1, ncols+1), dtype=np.uint8) + 1
xx,yy = disk((nrows/2,nrows/2),nrows/2+1)
z[xx,yy] = 1

xx,yy = disk((nrows/2,nrows/2),nrows/4)
z[xx,yy] = 20

xx,yy = disk((nrows/2,nrows/2),nrows/8)
z[xx,yy] = 100


#xx, yy = ellipse(nrows/2, nrows/3, nrows/4, nrows/6, rotati    on=np.deg2rad(30))
#z[xx,yy] = 1

#xx, yy = ellipse(nrows/5, nrows/3, nrows/6, nrows/4, rotation=np.deg2rad(10))
#z[xx,yy] = 1

#xx, yy = ellipse(60, 60, nrows/10, nrows/4, rotation=np.deg2rad(10))
#z[xx,yy] = 1


xx, yy, val = line_aa(0, 0, nrows, nrows)
z[xx,yy] = 1
xx, yy, val = line_aa(0, 2, nrows, nrows-2)
z[xx,yy] = 1
xx, yy, val = line_aa(2, 0, nrows-2, nrows)
z[xx,yy] = 1
z = np.asarray(z,dtype=np.double) 

print(z)



class ValidityChecker(ob.StateValidityChecker):
    
    # Returns whether the given state's position overlaps the
    # circular obstacle
    def isValid(self, state):
        #return True
        return self.clearance(state) > 0.0

    # Returns the distance from the given state's position to the
    # boundary of the circular obstacle.
    def clearance(self, state):
        # Extract the robot's (x,y) position from its state
        x = state[0]
        y = state[1]
        #z = state[2]
        # Distance formula between two points, offset by the circle's
        # radius
        return sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) - 0.05

    #Obstable structure: [(polygon,base,topo),(polygon,base,topo)]
    #def isValid(self,state):
    #    return self.clearence(state)

    #def clearence(self,state):
    #    x = state[0]
    #    y = state[1]
        #z = state[2]

    #    for polygon in obstacle:
    #        polygon_shp = Polygon(polygon[0])
    #        point_shp =  Point((x,y))
    #        if polygon_shp.contains(point_shp):
                #if z > polygon[1] or z < polygon[2]: 
    #                return False
                #else:
                #   return True
    #        else:
    #            return True
 




def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)


def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    #def stateCost(self, s):
    #    x = s[0]*z.shape[0]
    #    y = s[1]*z.shape[1]
    #    x = int(x)-1
    #    y = int(y)-1
    #    return ob.Cost(z[x][y])
    #    return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
    #                        sys.float_info.min))

    def motionCost(self,s1,s2):
        #print('State 1:'+ str(s1[0]),str(s1[1]))
        #print('State 2:'+ str(s2[0]),str(s2[1]))
        x1 = round(s1[0]*(z.shape[0]-1))
        y1 = round(s1[1]*(z.shape[1]-1))
        x2 = round(s2[0]*(z.shape[0]-1))
        y2 = round(s2[1]*(z.shape[1]-1))
        
        xx, yy, val = line_aa(x1, y1, x2, y2)
        #img = np.zeros((z.shape[0], z.shape[1]), dtype=np.uint8)
        #img[xx, yy] = val * 100
        #print(img)


        cost = 0
        for idx in range(len(xx)-1):
            cost = cost + z[xx[idx+1]][yy[idx+1]]*0.01
        return ob.Cost(cost)

def getClearanceObjective(si):
    return ClearanceObjective(si)


def getBalancedObjective1(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    lengthObj.setCostThreshold(ob.Cost(1.5))
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 1.0)
    opt.addObjective(clearObj, 1.0)

    return opt





def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
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
def allocateObjective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

def decodeSolutionPath(path, plannerType, pathCost):
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

    return (solution_lat,solution_lon,plannerType)

def plan(runTime, plannerType, objectiveType, fname):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.
    space = ob.RealVectorStateSpace(2)

    # Set the bounds of space to be in [0,1].
    space.setBounds(0.0, 1.0)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)
    
    # Set the object used to check which states in the space are valid
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)

    si.setup()

    # Set our robot's starting state to be the bottom-left corner of
    # the environment, or (0,0).
    start = ob.State(space)
    start[0] = 0.0
    start[1] = 0.0
    #start[2] = 0.0  

    # Set our robot's goal state to be the top-right corner of the
    # environment, or (1,1).
    goal = ob.State(space)
    goal[0] = 1.0
    goal[1] = 1.0
    #goal[2] = 1.0

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
        # Output the length of the path found
        print('{0} found solution of path length {1:.4f} with an optimization ' \
            'objective value of {2:.4f}'.format( \
            optimizingPlanner.getName(), \
            pdef.getSolutionPath().length(), \
            pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))
        
        return decodeSolutionPath(pdef.getSolutionPath().printAsMatrix(),plannerType, pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value())
        
       

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
    solution = []
    #for planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar']:
    for planner in [ 'RRTstar','InformedRRTstar','BITstar','PRMstar']:
        solution.append(plan(5, planner, 'WeightedLengthAndClearanceCombo', args.file))
    #PathClearance  WeightedLengthAndClearanceCombo
    import random
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt


 
    #data = np.loadtxt('path.txt')
    #fig, ax = plt.subplots()
    #ax = fig.gca(projection='3d')mult
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

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
        for sol in solution:
            ax.plot(sol[0],sol[1],label=sol[2])
    except:
        pass
    ax.set_title('image (nearest, aspect="auto")')
    #fig.colorbar(im, ax=ax)
    ax.set(xlabel='Latitude', ylabel='Longitude',
    title='Simple Plot')
    ax.legend()






    for sol in solution:
        x_main=[]
        y_main=[]
        x_points = []
        y_points = []
        v_value = []
        img = np.zeros((z.shape[0], z.shape[1]), dtype=np.uint8)
        try:
            for idx in range(len(sol[0])-1):
                x1 = round(sol[0][idx]*(z.shape[0]-1))
                y1 = round(sol[1][idx]*(z.shape[1]-1))
                x2 = round(sol[0][idx+1]*(z.shape[0]-1))
                y2 = round(sol[1][idx+1]*(z.shape[1]-1))
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
        ax.plot(x_points,y_points,'.')
        ax.plot(x_main,y_main)






    plt.show()
    
    
    ''' fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ims = []
    for i in range(60):
        
        art = []
        # make these smaller to increase the resolution
        dx, dy = 0.015, 0.005

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[0:1+dy:dy, 0:1+dx:dx]
        rand = random.random()
        z = (rand - x/rand + x**5 + y**3) * np.exp(-x**2 - y**2)
        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        z = z[:-1, :-1]
        z_min, z_max = -abs(z).max(), abs(z).max()

        art = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto', animated=True)
        if i == 0:
            ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto')
        #art.append(ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max))

        for sol in solution:
            art2 = ax.plot(sol[0],sol[1],label=sol[2])
            if i == 0:
                ax.plot(sol[0],sol[1],label=sol[2])
        #art.append(ax.set_title('image (nearest, aspect="auto")'))
        #fig.colorbar(im, ax=ax)
        #art.append(ax.set(xlabel='Latitude', ylabel='Longitude',title='Simple Plot'))
        #art.append(ax.legend())
        #ax.grid()

        
        ims.append([art2])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                    repeat_delay=1000)

    
    #ax.set_title('pcolor')
    #fig.colorbar(c, ax=ax)

    #nrows = 10
    #ncols = 10
    #Z =[[50,0,0,0,0,0,0,0,0,0,100],\
    #    [0,50,0,0,0,0,0,0,0,100,0],\
    #    [0,0,50,0,0,0,0,0,100,0,0],\
    #    [0,0,0,50,0,0,0,100,0,0,0],\
    #    [0,0,0,0,50,0,100,0,0,0,0],\
    #    [0,0,0,0,100,0,50,0,0,0,0],\
    #    [0,0,0,100,0,0,0,50,0,0,0],\
    #    [0,0,100,0,0,0,0,0,50,0,0],\
    #    [0,100,0,0,0,0,0,0,0,50,0],\
    #    [100,0,0,0,0,0,0,0,0,0,50],\
    #    [100,0,0,0,0,0,0,0,0,0,50]]
    #Z = np.asarray(Z) 
    #x = np.arange(ncols+1)*0.1
    #y = np.arange(nrows+1)*0.1
    #ax.pcolormesh(x, y, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())

    #X, Y = np.meshgrid(x, y)
    #ax.plot(X.flat, Y.flat, '.', color='m')
    
    
    plt.show() '''

    #for polygon in obstacle:
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

    
    #plt.show()

    #circle = plt.Circle((0.5,0.5),0.25)
    #ax.add_artist(circle)

    #N=20
    #stride=1
    #radius = 0.25
    #u = np.linspace(0, 2 * np.pi, N)
    #v = np.linspace(0, np.pi, N)
    #x = np.outer(np.cos(u), np.sin(v)) * radius + 0.5
    #y = np.outer(np.sin(u), np.sin(v)) * radius + 0.5
    #z = np.outer(np.ones(np.size(u)), np.cos(v)) * radius + 0.5
    #ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride)

