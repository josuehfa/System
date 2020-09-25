import subprocess
import os
import os.path
import sys
import time
import json
import signal
import requests
from BatchGSModule import *
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyredemet.src.pyredemet import *

#from CoreClass import *



def containsSTSC(point_list, polygon):
    '''Filter the data from STSC inside an specific polygon'''
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    points_in = []
    polygon_shp = Polygon(polygon)
    for _pnt in point_list:
        point = (float(_pnt['la']),float(_pnt['lo']))
        point_shp =  Point(point)
        if polygon_shp.contains(point_shp):
            points_in.append(point)
    return points_in


def showContains(point_list, polygon_list, location=[-11,-50], zoom_start=1, filepath='/home/josuehfa/System/CoreSystem/map.html'):
    '''Use Folium to plot an interative map with the constrained points'''
    import folium
    import webbrowser
   
    #Create a Map instance
    m = folium.Map(location=location,zoom_start=zoom_start,control_scale=True)
    for _pnt in point_list:
        folium.Circle(radius=100, location=[_pnt[0], _pnt[1]], color='crimson', fill=True).add_to(m)
    for _plg in polygon_list:
        folium.PolyLine(_plg, weight=2, color="blue").add_to(m)
    m.save(filepath)
    webbrowser.open('file://'+filepath)

def transPointsToPolygons(points, radius=0.05):
    '''Use the points from STSC to create polygon to be converted in geofences'''
    polygon_list = []
    for _pnt in points:
        x = _pnt[0]
        y = _pnt[1]
        polygon = [(x-radius,y+radius),(x+radius,y+radius),
                   (x+radius,y+radius),(x+radius,y-radius),
                   (x+radius,y-radius),(x-radius,y-radius),
                   (x-radius,y-radius),(x-radius,y+radius)]
        polygon_list.append(polygon)
    return polygon_list


def CreateGeofence(polygons):

    fenceList = []
    for idx, polygon in enumerate(polygons):
        id = idx
        type = 1 #Exclide
        numV = len(polygon)
        floor = 0
        roof = 100
        Vertices = []

        for vertex in polygon:
            coord = (vertex[0], vertex[1])
            Vertices.append(coord)

        Geofence = {'id': id, 'type': type, 'numV': numV, 'floor': floor,
                    'roof': roof, 'Vertices': Vertices}
        fenceList.append(Geofence)

    return fenceList

#Create a geofence class with ABC to be able to call the same class with differents contructors
def loadGeofence(polygons):
    '''load fence points '''
    fenceList = []
    try:
        fenceList = CreateGeofence(polygons)
    except Exception as msg:
        print("Unable to load %s - %s" % (polygons, msg))
        return
    return fenceList
    #for fence in fenceList:
    #    if fence not in sentFenceList:
    #        Send_fence(fence)


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
stsc_data = redemet.get_produto_stsc(data=data, anima=1)

#polygon = [(-5.00, -58.67), (-5.38, -39.33),
#            (-5.38, -39.33), (-18.09, -39.41),
#            (-18.09, -39.41), (-17.83, -53.82),
#            (-17.83, -53.82), (-5.00, -58.67)]
polygon = [(-12.0, -47.98), (-12.0, -46.99),
            (-12.0, -46.99), (-12.67, -46.99),
            (-12.67, -46.99), (-12.67, -47.98),
            (-12.67, -47.98), (-12.0, -47.98)]

points_in = containsSTSC(stsc_data['stsc'][0], polygon)
polygon_list = transPointsToPolygons(points_in, radius=0.05)
polygon_show = polygon_list + [polygon]
showContains(points_in, polygon_show, location=[-16,-50], zoom_start=6, filepath='/home/josuehfa/System/CoreSystem/map.html')
geolist = loadGeofence(polygon_list)
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
GS.loadGeofence(polygon_list, _type = 1)
#GS.loadGeofence("/home/josuehfa/System/icarous/Scripts/geofence2.xml")
GS.StartMission()
time.sleep(120)
input()
#os.kill(sim.pid,signal.SIGTERM)
#sim.kill()
#subprocess.Popen(["pkill","xterm"])
#sitl.kill()
#gs.kill()
#os.kill(os.getpid(),signal.SIGTERM)