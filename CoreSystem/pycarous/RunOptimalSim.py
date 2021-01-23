import sys, os
from SimEnvironment import SimEnvironment
from Icarous import Icarous
from IcarousRunner import IcarousRunner
from ichelper import GetHomePosition,ReadTrafficInput
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from OptimalClass import *
from MapGenClass import *
from PlotlyClass import *
from ScenarioClass import *
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse

def checkDAAType(value):
    if value.upper() not in ['DAIDALUS','ACAS']:
        raise argparse.ArgumentTypeError("%s is an invalid DAA option" % value)
    return value.upper()

parser = argparse.ArgumentParser(description=\
" Run a fast time simulation of Icarous with the provided inputs.\n\
  - See available input flags below.\n\
  - Icarous output are written to log file: SPEEDBIRD.json\n\
  - Use VisualizeLog.py to animate/record simoutput.\n\
  - See VisualizeLog.py --help for more information.",
                    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--daaType", type=checkDAAType, default='DAIDALUS', 
                   help='Specify DAA modules to be used. DAIDALUS,ACAS,...')
parser.add_argument("-f", "--flightplan", type=str, default='/home/josuehfa/System/CoreSystem/pycarous/data/flightplan.txt', 
                   help='flightplan file. default: data/flightplan.txt')
parser.add_argument("-t", "--traffic", type=str, default='/home/josuehfa/System/CoreSystem/pycarous/data/traffic.txt',
                   help='File containing traffic initial condition. See data/traffic.txt for example')
parser.add_argument("-l", "--tlimit", type=float, default=1000,
                   help='set max sim time limit (in seconds). default 300 s')
parser.add_argument("-p", "--params", type=str, default='/home/josuehfa/System/CoreSystem/pycarous/data/icarous_default.parm',
                   help='icarous parameter file. default: data/icarous_default.parm')
parser.add_argument("-g", "--geofence", type=str, default='',
                   help='geofence xml input. example: data/geofence2.xml')
parser.add_argument("-c", "--daaConfig", type=str, default='/home/josuehfa/System/CoreSystem/pycarous/data/DaidalusQuadConfig.txt',
                   help='specify configuration file if one is required by the DAA module specified by -d/--daaType')
parser.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=1,
                   help='Set print verbosity level')
parser.add_argument("--realtime", dest="fasttime", action="store_false",
                   help='Run sim in real time')
parser.add_argument("--fasttime", dest="fasttime", action="store_true",
                   help='Run sim in fast time (not available for cFS simulations)')
parser.add_argument("--cfs", action="store_true",
                   help='Run Icarous using cFS instead of pycarous')
parser.add_argument("-u", "--uncertainty", type=bool, default=False,
                   help='Enable uncertainty')
parser.add_argument("-r", "--repair", action="store_true",
                   help='Convert the given flightplan into a EUTL plan')
parser.add_argument("-e", "--eta", action="store_true",
                   help='Enable eta control for waypoint arrivals')
parser.add_argument("--daalog",  action="store_true",
                   help='Enable daa logs')
args = parser.parse_args()
if args.cfs:
    args.fasttime = False

#OptimalClass Startup
start_time = time.time()
scenario = ScenarioClass('FOUR')
dimension = '2D'
planner = 'RRTstar'
processing_time = 1

plans = []
fig = plt.figure()
axis = plt.axes(xlim =(-0.2, 1.2),ylim =(-0.2, 1.2))
delta_d = 1/scenario.nrows
ims = []
path_x = []
path_y = []
plan_aux = []
cost_aut = []

for idx, alg in enumerate(['RRTstar']):
    plan_aux.append(OptimalPlanning(scenario.start, scenario.goal, scenario.region, scenario.obstacle, planner, dimension, scenario.mapgen.z_time[round(0)]))
    result = plan_aux[idx].plan(10, alg, 'WeightedLengthAndClearanceCombo',delta_d)
    if plan_aux[idx].solution != []:
        cost_aut.append(plan_aux[idx].solution[0][3])
    else:
        cost_aut.append(np.inf)
lower_cost = cost_aut.index(min(cost_aut))

if plan_aux[lower_cost].solution != []:
    plans.append(plan_aux[lower_cost])
    PlanningStatus(scenario,plans)
    print('Start Solution')
    
    path_x.append(plans[-1].solution[0][0][0])
    path_y.append(plans[-1].solution[0][1][0])
    ims.append(plotResult(plans[-1],axis, scenario, path_x, path_y, 0))

fp_start = []
for idx in range(len(plans[-1].solution[0][0])):
    lat = plans[-1].solution[0][1][idx]*scenario.lat_range + min(scenario.lat_region)
    lon = plans[-1].solution[0][0][idx]*scenario.lon_range + min(scenario.lon_region)
    
    #[lat, lon, alt, wp_metric, tcp, tcp_value]
    #fp = [[37.102177, -76.387207, 5.0, 1.0, [0, 0, 0], [0.0, 0, 0]]]
    if idx == 0 or idx == len(plans[-1].solution[0][0])-1:
        fp_start.append([lat, lon, 0.0, 4.5, [0, 0, 0], [0.0, 0, 0]])
    else:
        fp_start.append([lat, lon, 5.0, 4.5, [0, 0, 0], [0.0, 0, 0]])


# Initialize simulation environment
sim = SimEnvironment(fasttime=args.fasttime,verbose=args.verbosity)

# Set the home position for the simulation
HomePos = list(scenario.start_real)

# Add traffic inputs
if args.traffic != '':
    tfinputs = ReadTrafficInput(args.traffic)
    for tf in tfinputs:
        sim.AddTraffic(tf[0], fp_start[1][:3], *tf[1:])

# Initialize Icarous class
if args.cfs:
    ic = IcarousRunner(HomePos, verbose=args.verbosity)
else:
    ic = Icarous(HomePos,simtype="UAM_VTOL",monitor=args.daaType,verbose=args.verbosity,
                daaConfig=args.daaConfig, fasttime=args.fasttime)

if args.daalog:
    # Dirty hack to silently update the daa logging parameter from commandline
    import os
    os.system("sed -Ein -e \'s/(LOGDAADATA)(\\ *)([0-1])(\\.0*)/\\1\\21\\4/\' "+args.params)

# Read params from file and input params
ic.SetParametersFromFile(args.params)

#Input flightplan
#Use the first plan of OMPL
ic.InputFlightplan(fp_start[:2],eta=args.eta,repair=args.repair)

# Input geofences from file
if args.geofence != '':
    ic.InputGeofence("data/geofence2.xml")

# Add icarous instance to sim environment
sim.AddIcarousInstance(ic,time_limit=args.tlimit)

# Set position uncertainty for vehicles in the simulation
if args.uncertainty:
    sim.SetPosUncertainty(0.1, 0.1, 0, 0, 0, 0)

# Run the Simulation
sim.RunSimulationOptimal(scenario, plans)

# Save json log outputs
sim.WriteLog()




print(str(time.time() - start_time) + ' seconds')
from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot

im_ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
#anim = FuncAnimation(fig, animate, init_func = init, 
#                    frames =len(path_y) , interval = 200, blit = True) 

# Set up formatting for the movie files
writermp4 = animation.FFMpegWriter(fps=60) 
#im_ani.save('CoreSystem/Results/path.mp4', writer=writermp4)

plt.show()