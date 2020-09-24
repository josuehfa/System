import subprocess
import os
import os.path
import sys
import time
import json
import signal
import requests
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyredemet.src.pyredemet import *
from BatchGSModule import *
from CoreClass import *



# Start sim vehicle simulation script
#sim = subprocess.Popen(["/home/josuehfa/System/ardupilot/Tools/autotest/sim_vehicle.py","-v","ArduCopter","-l","37.1021769,-76.3872069,5,0","-S","1"],stdout=subprocess.PIPE , shell=True)
#time.sleep(30)

# Start ICAROUS
#sitl = subprocess.Popen(['cd','/home/josuehfa/System/icarous/exe/cpu1/','&&','./core-cpu1', '-C', '1', '-I', '0'], shell=True, stdout=subprocess.PIPE)
#time.sleep(30)

#Start MAVProxy
#gs = subprocess.Popen(['cd','/home/josuehfa/System/icarous/Scripts/','&&','./runGS.sh'], shell=True, stdout=subprocess.PIPE)
#time.sleep(30)

with open(os.path.join(os.path.dirname(__file__),'config.json')) as json_file:
    data = json.load(json_file)


api_key = data["api_key"]
redemet = pyredemet(api_key)

data = '2020092322'
anima=1
stsc_data = redemet.get_produto_stsc(data=data, anima=anima)

Pt = namedtuple('Pt', 'x, y')               # Point
Edge = namedtuple('Edge', 'a, b')           # Polygon edge from a to b
Poly = namedtuple('Poly', 'name, edges')    # Polygon

poly = Poly(name='geofance', edges=(
            Edge(a=Pt(x=-5.00, y=-58.67), b=Pt(x=-5.38, y=-39.33)),
            Edge(a=Pt(x=-5.38, y=-39.33), b=Pt(x=-18.09, y=-39.41)),
            Edge(a=Pt(x=-18.09, y=-39.41), b=Pt(x=-17.83, y=-53.82)),
            Edge(a=Pt(x=-17.83, y=-53.82), b=Pt(x=-5.00, y=-58.67))
            ))
stsc_check = []
for idx, data in enumerate(stsc_data['stsc'][0]):
    point = Pt(x=float(data['la']), y=float(data['lo']))
    if ispointinside(point, poly):
        stsc_check.append(point)

import os
import webbrowser
import folium
#Create a Map instance
m = folium.Map(location=[-11,-50],zoom_start=10,control_scale=True)
for point in stsc_check:
    folium.Circle(radius=100, location=[point.x,point.y],color='crimson',fill=False).add_to(m)

polygon = [[-5.00, -58.67], [-5.38, -39.33],
            [-5.38, -39.33], [-18.09, -39.41],
            [-18.09, -39.41], [-17.83, -53.82],
            [-17.83, -53.82], [-5.00, -58.67]]

folium.PolyLine(polygon, weight=2, color="blue").add_to(m)

filepath = '/home/josuehfa/System/CoreSystem/map.html'

m.save(filepath)
webbrowser.open('file://'+filepath)

pais = 'Brasil'
data_ini = '202007071200'
data_fim = '202007071800'
sigmet_data = redemet.get_mensagens_sigmet(pais=pais, data_ini=data_ini, data_fim=data_fim)
# radius 0.1 deg

def decoder_sigmet(sigmet_data):
    sigmet_list = {}
    for idx in range(0,len(sigmet_data['data'])):
        if idx < 10: idx = '0'+str(idx) #Put a 0 in the begginer, redemet protocol
        else: idx = str(idx)
        lat_lon = {"lat_lon":sigmet_data['data'][idx]['lat_lon']}

        flight_level = sigmet_data['data'][idx]['fenomeno_comp']
        
        
        
        
        #sigmet.append({lat_lon:sigmet_data['data'][idx]},()}






# Open a mavlink UDP port
master = None
try:
    master = mavutil.mavlink_connection("udp:127.0.0.1:14553", source_system=1)
    #print (master)
except Exception as msg:
    print ("Error opening mavlink connection " + str(msg))

master.wait_heartbeat()
GS = BatchGSModule(master,1,0)
GS.loadWaypoint("/home/josuehfa/System/icarous/Scripts/flightplan4.txt")
GS.loadGeofence("/home/josuehfa/System/icarous/Scripts/geofence2.xml")
GS.StartMission()
time.sleep(120)
input()
#os.kill(sim.pid,signal.SIGTERM)
#sim.kill()
#subprocess.Popen(["pkill","xterm"])
#sitl.kill()
#gs.kill()
#os.kill(os.getpid(),signal.SIGTERM)