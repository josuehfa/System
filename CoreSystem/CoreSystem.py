
import os
import os.path
import sys
import time
import json
import signal
import requests
import subprocess
import threading
from BatchGSModule import *
from CoreClass import *
from PlanningClass import *
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyredemet.src.pyredemet import *

#date = '2020092512'
#polygon = [(-5.00, -58.67), (-5.38, -39.33),
#            (-5.38, -39.33), (-18.09, -39.41),
#            (-18.09, -39.41), (-17.83, -53.82),
#            (-17.83, -53.82), (-5.00, -58.67)]


# def execute(cmd):
#     sim = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
#     for stdout_line in iter(sim.stdout.readline, ""):
#         yield stdout_line 
#     sim.stdout.close()
#     return_code = sim.wait()
#     if return_code:
#         raise subprocess.CalledProcessError(return_code, cmd)

#Start Ardupilot
#cmd = ["/home/josuehfa/System/ardupilot/Tools/autotest/sim_vehicle.py","-v","ArduCopter","-l","-12.576857, -47.773923, 5, 0"]
#x = threading.Thread(target=execute, args=(cmd,))
#x.start()
#time.sleep(10)
#print('Ardupilot Loaded - Start ICAROUS')

#(-12.576857, -47.773923)
_type = 'stsc' 
#date = '2020092322'
date = '2020092323'
polygon = [(-12.0, -47.98),
            (-12.0, -46.99), 
            (-12.67, -46.99), 
            (-12.67, -47.98)]

redemet = RedemetCore()
polygon_list = redemet.getPolygons(_type, date, polygon)
#redemet.showPolygons(polygon)


#polygon = [(-3, -2),
#            (-3, 5), 
#            (6, 5),
#            (6, -2)]
base = 1
topo = 10

#polygon2 = [(8, 9),
#            (8, 6),
#            (6, 6),
#            (6, 9)]
obstacle = []
for pol in polygon_list:
    
    obstacle.append((pol, base, topo))
#print(obstacle)

start =(-12.625412, -47.867092, 1) 
goal = (-12.214918, -47.283527, 8)
region = [(-12.0, -47.98), 
            (-12.0, -46.99), 
            (-12.67, -46.99), 
            (-12.67, -47.98)]
#redemet.showPolygons(polygon=polygon,points=[start,goal])

dimension = '3D'
planner = 'RRTstar'

plan = PathPlanning(start, goal, region, obstacle, planner, dimension)
#result = plan.plan(10, planner, 'PathLength')


for planner in ['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', 'SORRTstar']:
    result = plan.plan(10, planner, 'PathLength')
redemet.showPolygons(polygon=polygon, lines=plan.solution,points=[start[0:2],goal[0:2]])
plan.plotSolutionPath(anima=False)

input()




# Open a mavlink UDP port
master = None
try:
    master = mavutil.mavlink_connection("udp:127.0.0.1:14553", source_system=1)
    #print (master)
except Exception as msg:
    print ("Error opening mavlink connection " + str(msg))

master.wait_heartbeat()
GS = BatchGSModule(master,1,0)
#GS.loadWaypoint("/home/josuehfa/System/icarous/Scripts/flightplan4.txt")
GS.loadGeofence(polygon_list, 1)
GS.loadGeofence([polygon], 0)
print('Geofence Loaded - Start WEBGS')
#GS.loadGeofence("/home/josuehfa/System/icarous/Scripts/geofence2.xml")
GS.StartMission()
time.sleep(30)
input()





# Start ICAROUS
#sitl = subprocess.Popen(['cd','/home/josuehfa/System/icarous/exe/cpu1/','&&','./core-cpu1', '-C', '1', '-I', '0'], shell=True, stdout=subprocess.PIPE)
#time.sleep(30)

#Start MAVProxy
#gs = subprocess.Popen(['cd','/home/josuehfa/System/icarous/Scripts/','&&','./runGS.sh'], shell=True, stdout=subprocess.PIPE)
#time.sleep(30)




#os.kill(sim.pid,signal.SIGTERM)
#sim.kill()
#subprocess.Popen(["pkill","xterm"])
#sitl.kill()
#gs.kill()
#os.kill(os.getpid(),signal.SIGTERM)
# radius 0.1 deg

#pais = 'Brasil'
#data_ini = '202007071200'
#data_fim = '202007071800'
#sigmet_data = redemet.get_mensagens_sigmet(pais=pais, data_ini=data_ini, data_fim=data_fim)


#def decoder_sigmet(sigmet_data):
#    sigmet_list = {}
#    for idx in range(0,len(sigmet_data['data'])):
#        if idx < 10: idx = '0'+str(idx) #Put a 0 in the begginer, redemet protocol
#        else: idx = str(idx)
#        lat_lon = {"lat_lon":sigmet_data['data'][idx]['lat_lon']}
#        flight_level = sigmet_data['data'][idx]['fenomeno_comp']
        

        
        #sigmet.append({lat_lon:sigmet_data['data'][idx]},()}
