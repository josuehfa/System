import os, time, platform
import xml.etree.ElementTree as ET
from pymavlink import mavutil, mavwp
import pymavlink

from pymavlink.dialects.v10 import icarous

import numpy as np
import math
from numpy import sin,cos

radius_of_earth = 6378100.0

def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1

    a = math.sin(0.5*dLat)**2 + math.sin(0.5*dLon)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    return radius_of_earth * c


def wrap_valid_longitude(lon):
    ''' wrap a longitude value around to always have a value in the range
        [-180, +180) i.e 0 => 0, 1 => 1, -1 => -1, 181 => -179, -181 => 179
    '''
    return (((lon + 180.0) % 360.0) - 180.0)

def gps_newpos(lat, lon, bearing, distance):
    '''extrapolate latitude/longitude given a heading and distance
    thanks to http://www.movable-type.co.uk/scripts/latlong.html
    '''
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing)
    dr = distance/radius_of_earth

    lat2 = math.asin(math.sin(lat1)*math.cos(dr) +
                     math.cos(lat1)*math.sin(dr)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), wrap_valid_longitude(math.degrees(lon2)))

def gps_offset(lat, lon, east, north):
    '''return new lat/lon after moving east/north
    by the given number of meters'''
    bearing = math.degrees(math.atan2(east, north))
    distance = math.sqrt(east**2 + north**2)
    return gps_newpos(lat, lon, bearing, distance)

def LLA2NED(origin,position):
        """
        Convert from geodetic coordinates to NED coordinates
        :param origin:  origin of NED frame in geodetic coordinates
        :param position: position to be converted to NED
        :return: returns position in NED
        """
        R    = 6371000  # radius of earth
        oLat = origin[0]*np.pi/180
        oLon = origin[1]*np.pi/180

        if(len(origin) > 2):
            oAlt = origin[2]
        else:
            oAlt = 0

        pLat = position[0]*np.pi/180
        pLon = position[1]*np.pi/180

        if(len (origin) > 2):
            pAlt = position[2]
        else:
            pAlt = 0

        # convert given positions from geodetic coordinate frame to ECEF
        oX   = (R+oAlt)*cos(oLat)*cos(oLon)
        oY   = (R+oAlt)*cos(oLat)*sin(oLon)
        oZ   = (R+oAlt)*sin(oLat)

        Pref = np.array([[oX],[oY],[oZ]])

        pX   = (R+pAlt)*cos(pLat)*cos(pLon)
        pY   = (R+pAlt)*cos(pLat)*sin(pLon)
        pZ   = (R+pAlt)*sin(pLat)

        P    = np.array([[pX],[pY],[pZ]])

        # Convert from ECEF to NED
        Rne  = np.array([[-sin(oLat)*cos(oLon), -sin(oLat)*sin(oLon), cos(oLat)],
                         [-sin(oLon),                cos(oLon),          0     ],
                         [-cos(oLat)*cos(oLon), -cos(oLat)*sin(oLon),-sin(oLat)]])

        Pn   = np.dot(Rne,(P - Pref))

        if(len (origin) > 2):
            return [Pn[0,0], Pn[1,0], Pn[2,0]]
        else:
            return [Pn[0,0], Pn[1,0]]


class Traffic:
    def __init__(self,distH,bearing,z,S,H,V,tstart):
        self.x0 = distH*math.sin((90 - bearing)*3.142/180);
        self.y0 = distH*math.cos((90 - bearing)*3.142/180);
        self.z0 = z;

        self.vx0 = S*math.sin((90 - H)*3.142/180);
        self.vy0 = S*math.cos((90 - H)*3.142/180);
        self.vz0 = V;

        self.x = self.x0;
        self.y = self.y0;
        self.z = self.z0;

        self.lat = 0
        self.lon = 0
        self.alt = 0

        self.tstart = tstart

    def get_pos(self,t):
        dt = t-self.tstart

        self.x = self.x0 + self.vx0*(dt)
        self.y = self.y0 + self.vy0*(dt)
        self.z = self.z0 + self.vz0*(dt)

class BatchGSModule():
    def __init__(self, master,target_system,target_component):
        self.master = master
        self.fenceList = []
        self.sentFenceList = []
        self.fenceToSend = 0
        self.have_list = False
        self.wploader = mavwp.MAVWPLoader()
        self.wp_op = None
        self.wp_requested = {}
        self.wp_received = {}
        self.wp_save_filename = None
        self.wploader = mavwp.MAVWPLoader()
        self.loading_waypoints = False
        self.loading_waypoint_lasttime = time.time()
        self.last_waypoint = 0
        self.wp_period = mavutil.periodic_event(0.5)
        self.target_system = target_system
        self.target_component = target_component
        self.traffic_list = []
        self.start_lat = 0
        self.start_lon = 0

    def loadGeofence(self, filename):
        '''load fence points from a file'''
        try:
            self.GetGeofence(filename)
        except Exception as msg:
            print("Unable to load %s - %s" % (filename, msg))
            return

        for fence in self.fenceList:
            if fence not in self.sentFenceList:
                self.Send_fence(fence)

    def StartMission(self):
        self.master.mav.command_long_send(self.target_system, self.target_component,
                                          mavutil.mavlink.MAV_CMD_MISSION_START,
                                          0, 0, 0, 0, 0, 0, 0, 0)

    def Send_fence(self, fence):
        '''send fence points from fenceloader'''
        target_system = 2
        target_component = 0
        self.master.mav.command_long_send(target_system, target_component,
                                          mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE, 0,
                                          0, fence["id"], fence["type"], fence["numV"],
                                          fence["floor"], fence["roof"], 0)

        fence_sent = False

        while (not fence_sent):

            msg = None
            while (msg == None):
                msg = self.master.recv_match(blocking=True, type=["FENCE_FETCH_POINT", "COMMAND_ACK"], timeout=50)

            if (msg.get_type() == "FENCE_FETCH_POINT"):
                print("received fetch point")
                numV = fence["numV"]
                lat = fence["Vertices"][msg.idx][0]
                lon = fence["Vertices"][msg.idx][1]

                self.master.mav.fence_point_send(1, 0, msg.idx, numV, lat, lon)


            elif (msg.get_type() == "COMMAND_ACK"):
                if msg.result == 0:
                    fence_sent = True
                    print("Geofence sent")
                else:
                    self.Send_fence(fence)
                    fence_sent = True

        points = fence["Vertices"][:]
        points.append(points[0])
        return

    def GetGeofence(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()

        for child in root:
            id = int(child.get('id'))
            type = int(child.find('type').text)
            numV = int(child.find('num_vertices').text)
            floor = float(child.find('floor').text)
            roof = float(child.find('roof').text)
            Vertices = []

            if (len(child.findall('vertex')) == numV):
                for vertex in child.findall('vertex'):
                    coord = (float(vertex.find('lat').text),
                             float(vertex.find('lon').text))

                    Vertices.append(coord)

                Geofence = {'id': id, 'type': type, 'numV': numV, 'floor': floor,
                            'roof': roof, 'Vertices': Vertices}

                self.fenceList.append(Geofence)

    def loadWaypoint(self,filename):
        '''load waypoints from a file'''
        try:
            self.wploader.load(filename)
        except Exception as msg:
            print("Unable to load %s - %s" % (filename, msg))
            return
        print("Loaded %u waypoints from %s" % (self.wploader.count(), filename))
        self.send_all_waypoints()

    def send_all_waypoints(self):
        '''send all waypoints to vehicle'''
        self.wploader.target_system = self.target_system
        self.wploader.target_component = self.target_component
        self.master.waypoint_clear_all_send()
        if self.wploader.count() == 0:
            return
        self.loading_waypoints = True
        self.loading_waypoint_lasttime = time.time()
        self.master.waypoint_count_send(self.wploader.count())
        while self.loading_waypoints:
            reqd_msgs = ['WAYPOINT_COUNT', 'MISSION_COUNT',
                         'WAYPOINT', 'MISSION_ITEM',
                         'WAYPOINT_REQUEST', 'MISSION_REQUEST'
                         'MISSION_ACK']
            data = self.master.recv_msg()
            if data is not None:
                self.mavlink_packet_wp(data)

    def missing_wps_to_request(self):
        ret = []
        tnow = time.time()
        next_seq = self.wploader.count()
        for i in range(5):
            seq = next_seq + i
            if seq + 1 > self.wploader.expected_count:
                continue
            if seq in self.wp_requested and tnow - self.wp_requested[seq] < 2:
                continue
            ret.append(seq)
        return ret

    def send_wp_requests(self, wps=None):
        '''send some more WP requests'''
        if wps is None:
            wps = self.missing_wps_to_request()
        tnow = time.time()
        for seq in wps:
            # print("REQUESTING %u/%u (%u)" % (seq, self.wploader.expected_count, i))
            self.wp_requested[seq] = tnow
            self.master.waypoint_request_send(seq)

    def process_waypoint_request(self, m, master):
        '''process a waypoint request from the master'''
        if (not self.loading_waypoints or
                    time.time() > self.loading_waypoint_lasttime + 10.0):
            self.loading_waypoints = False
            print("not loading waypoints")
            return
        if m.seq >= self.wploader.count():
            print("Request for bad waypoint %u (max %u)" % (m.seq, self.wploader.count()))
            return
        wp = self.wploader.wp(m.seq)
        wp.target_system = self.target_system
        wp.target_component = self.target_component
        self.master.mav.send(self.wploader.wp(m.seq))
        self.loading_waypoint_lasttime = time.time()
        #print("Sent waypoint %u : %s" % (m.seq, self.wploader.wp(m.seq)))
        if m.seq == self.wploader.count() - 1:
            self.loading_waypoints = False
            #print("Sent all %u waypoints" % self.wploader.count())

    def loadParams(self, filename):
        '''load parameters from a file'''
        print("Loading params %s" % filename)
        try:
            f = open(filename, mode='r')
        except (IOError, TypeError):
            print("Failed to open file '%s'" % filename)
            return

        for line in f:
            line = line.replace('=',' ')
            line = line.strip()
            if not line or line[0] == "#":
                continue
            a = line.split()
            if len(a) < 1:
                print("Invalid line: %s" % line)
                continue
            self.master.param_set_send(a[0],float(a[1]))

    def setParam(self,param_id,param_value):
        '''set an individual parameter'''
        self.master.param_set_send(param_id,param_value)

    def getParams(self):
        '''request parameters'''
        params = {}
        t0 = time.time()
        self.master.param_fetch_all()
        while time.time() - t0 < 2:
            msg = self.master.recv_match(type="PARAM_VALUE")
            if msg != None:
                #print(msg.param_id,msg.param_value)
                params[msg.param_id] = msg.param_value
        return params

    def mavlink_packet_wp(self, m):
        '''handle an incoming mavlink packet'''
        mtype = m.get_type()
        if mtype in ['WAYPOINT_COUNT', 'MISSION_COUNT']:
            if self.wp_op is None:
                print("No waypoint load started")
            else:
                self.wploader.clear()
                self.wploader.expected_count = m.count
                print("Requesting %u waypoints t=%s now=%s" % (m.count,
                                                                              time.asctime(
                                                                                  time.localtime(m._timestamp)),
                                                                              time.asctime()))
                self.send_wp_requests()

        elif mtype in ['WAYPOINT', 'MISSION_ITEM'] and self.wp_op != None:
            if m.seq < self.wploader.count():
                # print("DUPLICATE %u" % m.seq)
                return
            if m.seq + 1 > self.wploader.expected_count:
                print("Unexpected waypoint number %u - expected %u" % (m.seq, self.wploader.count()))
            self.wp_received[m.seq] = m
            next_seq = self.wploader.count()
            while next_seq in self.wp_received:
                m = self.wp_received.pop(next_seq)
                self.wploader.add(m)
                next_seq += 1
            if self.wploader.count() != self.wploader.expected_count:
                # print("m.seq=%u expected_count=%u" % (m.seq, self.wploader.expected_count))
                self.send_wp_requests()
                return
            if self.wp_op == 'list':
                for i in range(self.wploader.count()):
                    w = self.wploader.wp(i)
                    print("%u %u %.10f %.10f %f p1=%.1f p2=%.1f p3=%.1f p4=%.1f cur=%u auto=%u" % (
                        w.command, w.frame, w.x, w.y, w.z,
                        w.param1, w.param2, w.param3, w.param4,
                        w.current, w.autocontinue))
            self.wp_op = None
            self.wp_requested = {}
            self.wp_received = {}

        elif mtype in ["WAYPOINT_REQUEST", "MISSION_REQUEST"]:
            self.process_waypoint_request(m, self.master)

        elif mtype in ["WAYPOINT_CURRENT", "MISSION_CURRENT"]:
            if m.seq != self.last_waypoint:
                self.last_waypoint = m.seq


    def load_traffic(self,args):
        start_time = time.time();
        tffc = Traffic(float(args[1]),float(args[2]),float(args[3]), \
                       float(args[4]),float(args[5]),float(args[6]),start_time)
        self.traffic_list.append(tffc)
        self.start_lat = self.wploader.wp(0).x
        self.start_lon = self.wploader.wp(0).y

    def Update_traffic(self):
        '''Update traffic icon on map'''

        #from MAVProxy.modules.mavproxy_map import mp_slipmap
        t = time.time()

        for i,tffc in enumerate(self.traffic_list):

            self.traffic_list[i].get_pos(t)
            (lat, lon) = gps_offset(self.start_lat,self.start_lon, self.traffic_list[i].y, self.traffic_list[i].x)
            self.traffic_list[i].lat = lat;
            self.traffic_list[i].lon = lon;
            self.traffic_list[i].alt = self.traffic_list[i].z
            heading = math.degrees(math.atan2(self.traffic_list[i].vy0, self.traffic_list[i].vx0))
            pos = [lat, lon, self.traffic_list[i].z0]
            vel = [self.traffic_list[i].vx0,
                   self.traffic_list[i].vz0,
                   self.traffic_list[i].vy0]
            self.Send_traffic(i, pos, vel)

    def Send_traffic(self, idx, position, velocity):
        '''
        Send traffic position updates to ICAROUS
        Input traffic surveillance data to ICAROUS
        :param idx: ID of the traffic vehicle
        :param position: traffic position [lat, lon, alt] (deg, deg, m)
        :param velocity: traffic velocity [vn, ve, vd] (m/s, m/s, m/s)
        '''
        self.master.mav.command_long_send(
            self.target_system,    # target_system
            self.target_component, # target_component
            mavutil.mavlink.MAV_CMD_SPATIAL_USER_1, # command
            0,           # confirmation
            idx,         # param1 (traffic id)
            velocity[0], # param2 (vn)
            velocity[1], # param3 (ve)
            velocity[2], # param4 (vd)
            position[0], # param5 (lat)
            position[1], # param6 (lon)
            position[2]) # param7 (alt)
