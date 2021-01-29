#! /usr/bin/env python3

import json
import os
import glob
import numpy as np
from Icarous import VisualizeSimData,VisualizeSimDataOptimal
from OptimalClass import *
from MapGenClass import *
from PlotlyClass import *
from ScenarioClass import *

class playback():
    def __init__(self):
        self.ownshipLog = []
        self.trafficLog = []
        self.localCoords = []
        self.localPlans = []
        self.localFences = []
        self.localMergeFixes = []
        self.daa_radius = []
        self.params = {}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Icarous log")
    parser.add_argument("--logfile", type=str, default="simlog-SPEEDBIRD.json",help="Icarous json log file or directory containing multiple json logs")
    parser.add_argument("--allplans", action="store_true", help="plot all planned paths")
    parser.add_argument("--notraffic", action="store_true", help="only show icarous vehicles")
    parser.add_argument("--record", action="store_true", help="record animation to file")
    parser.add_argument("--output", default="animation.mp4", help="video file name with .mp4 extension")
    parser.add_argument("--pad",type=float, default=0.0, help="extend the min/max values of the axes by the padding (in meters), default = 25.0 [m]")
    parser.add_argument("--speed",type=int, default=1.0, help="increase playback speed by given factor")
    args = parser.parse_args()

    files = []
    pbs   = []
    if os.path.isfile(args.logfile):
        files.append(args.logfile)
    else:
        path =  args.logfile.rstrip('/')+'/*.json'
        files = glob.glob(path)

    xmin, ymin = 1e10, 1e10
    xmax, ymax = -1e10, -1e10 
    valid = False
    for file in files:
        try:
            fp = open(file,'r')
            data = json.load(fp)
            valid = True
            pb = playback()
            pb.ownshipLog = data['ownship']
            pb.trafficLog = data['traffic']
            pb.localPlans = pb.ownshipLog['localPlans']
            pb.localCoords = pb.ownshipLog['localCoords'] 
            pb.localFences = pb.ownshipLog['localFences']
            pb.params = data['parameters']
            pb.daa_radius = pb.params['DET_1_WCV_DTHR']*0.3048*0.00001
            pb.localMergeFixes = data['mergefixes']
            scenario_time = data['scenario_time']
            scenario = ScenarioClass(data['scenario'])
            pbs.append(pb)
            #_xmin = np.min(np.array(pb.ownshipLog['position'])[:,1])
            #_xmax = np.max(np.array(pb.ownshipLog['position'])[:,1])
            #_ymin = np.min(np.array(pb.ownshipLog['position'])[:,0])
            #_ymax = np.max(np.array(pb.ownshipLog['position'])[:,0])
            #_xminfp = np.min(np.array(pb.localCoords)[:,1])
            #_xmaxfp = np.max(np.array(pb.localCoords)[:,1])
            #_yminfp = np.min(np.array(pb.localCoords)[:,0])
            #_ymaxfp = np.max(np.array(pb.localCoords)[:,0])
            #_xmin = np.min([_xmin,_xminfp])
            #_xmax = np.max([_xmax,_xmaxfp])
            #_ymin = np.min([_ymin,_yminfp])
            #_ymax = np.max([_ymax,_ymaxfp])
            #xmin = np.min([xmin,_xmin])
            #ymin = np.min([ymin,_ymin])
            #xmax = np.max([xmax,_xmax])
            #ymax = np.max([ymax,_ymax])
        except:
            continue

    if valid:
        #if (xmax-xmin) > (ymax-ymin):
        #     ymin = ymin + (ymax - ymin)/2 - (xmax-xmin)/2
        #     ymax = ymin + (xmax - xmin)
        # elif (ymax-ymin) > (xmax-xmin):
        #     xmin = xmin + (xmax - xmin)/2 - (ymax-ymin)/2
        #     xmax = xmin + (ymax - ymin)

         #padding = args.pad
        # xmin -= xmin*0.05
        # ymin -= ymin*0.05
        # xmax += xmax*0.05
        # ymax += ymax*0.05
        #VisualizeSimData(pbs,allplans=args.allplans,showtraffic=not args.notraffic,xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,playbkspeed=args.speed,interval=5,record=args.record,filename=args.output)
        lat_y,lon_x = zip(*scenario.region_real)
        xmax = max(lon_x)
        xmin = min(lon_x)
        ymax = max(lat_y)
        ymin = min(lat_y)
        VisualizeSimDataOptimal(pbs,scenario_time, scenario,allplans=args.allplans,showtraffic=not args.notraffic,xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,playbkspeed=args.speed,interval=5,record=args.record,filename=args.output)

    

