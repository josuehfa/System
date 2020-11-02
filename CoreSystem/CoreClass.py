#!/usr/bin/ebv python
# coding=utf-8

import os
import os.path
import sys
import time
import json
import folium
import webbrowser
from multipledispatch import dispatch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pyredemet.src.pyredemet import *

class RedemetCore():
    def __init__(self, config='config.json'):
        '''Class Constructor'''
        self.point_list = []
        self.polygon_list = []
        #Get API key
        with open(os.path.join(os.path.dirname(__file__),config)) as json_file:
            data = json.load(json_file)
        api_key = data["api_key"]
        self.redemet = pyredemet(api_key)

    @dispatch(str,int)
    def decoderSTSC(self, date_ini, anima):
        '''Return all points of STSC'''
        stsc_data = self.redemet.get_produto_stsc(date_ini, anima)
        stsc_points = stsc_data['stsc'][0]
        for _pnt in stsc_points:
            point = (float(_pnt['la']),float(_pnt['lo']))
            self.point_list.append(point)

    @dispatch(str,list,int)
    def decoderSTSC(self, date_ini, polygon, anima):
        '''Filter the data from STSC inside an specific polygon'''
        stsc_data = self.redemet.get_produto_stsc(date_ini, anima)
        stsc_points = stsc_data['stsc'][0]
        polygon_shp = Polygon(polygon)
        for _pnt in stsc_points:
            point = (float(_pnt['la']),float(_pnt['lo']))
            point_shp =  Point(point)
            if polygon_shp.contains(point_shp):
                self.point_list.append(point)
    
    def showPolygons(self, polygon=[], location=[-11,-50], zoom_start=5, filepath='/home/josuehfa/System/CoreSystem/map.html'):
        '''Use Folium to plot an interative map with points and polygons'''
        #Create a Map instance
        m = folium.Map(location=location,zoom_start=zoom_start,control_scale=True)
        for _pnt in self.point_list:
            folium.Circle(radius=100, location=[_pnt[0], _pnt[1]], color='crimson', fill=True).add_to(m)
        for _plg in self.polygon_list:
            folium.PolyLine(_plg, weight=2, color="blue").add_to(m)
        if polygon != []:
            folium.PolyLine(polygon, weight=2, color="blue").add_to(m)
        m.save(filepath)
        webbrowser.open('file://'+filepath)

    def transPointsToPolygons(self, radius=0.05):
        '''Use the points from RedeMet to create polygon to be converted in geofences'''
        for _pnt in self.point_list:
            x = _pnt[0]
            y = _pnt[1]
            polygon = [(x-radius,y+radius),(x+radius,y+radius),
                    (x+radius,y+radius),(x+radius,y-radius),
                    (x+radius,y-radius),(x-radius,y-radius),
                    (x-radius,y-radius),(x-radius,y+radius)]
            self.polygon_list.append(polygon)
    
    @dispatch(str,str)
    def getPolygons(self, _type, date_ini):
        '''Return a polygon list of STSC data'''
        if _type == 'stsc':
            self.decoderSTSC(date_ini, 1)
            self.transPointsToPolygons(radius=0.05)
        elif _type == 'sigmet':
            pass
        elif _type == 'stsc-sigmet':
            pass
        else:
            print ("Enter with a valid type of request")

    @dispatch(str,str,list)
    def getPolygons(self, _type, date_ini, polygon):
        '''Return a polygon list of STSC data constranied by a polygon'''
        if _type == 'stsc':
            self.decoderSTSC(date_ini, polygon, 1)
            self.transPointsToPolygons(radius=0.05)
        elif _type == 'sigmet':
            pass
        elif _type == 'stsc-sigmet':
            pass
        else:
            print ("Enter with a valid type of request")
        
        return self.polygon_list

