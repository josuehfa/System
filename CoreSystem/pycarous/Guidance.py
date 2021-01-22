from ctypes import *
from enum import IntEnum
from Interfaces import Waypoint

class GuidanceMode(IntEnum):
    FLIGHTPLAN = 0
    VECTOR = 1
    POINT2POINT = 2
    TAKEOFF = 3
    LAND = 4
    NOOP = 5

class GuidanceOutput(Structure):
     _fields_=[
          ("activePlan",c_char*25),
          ("guidanceMode",c_int),
          ("nextWP",c_int),
          ("distH2WP",c_double),
          ("distV2WP",c_double),
          ("xtrackDev",c_double),
          ("velCmd",c_double*3),
          ("target",c_double*3),
          ("wpReached",c_bool)
     ]

class GuidanceParam(Structure):
     _fields_ = [ \
         ( "defaultWpSpeed",c_double ), \
         ( "captureRadiusScaling",c_double ), \
         ( "guidanceRadiusScaling",c_double ), \
         ( "xtrkDev",c_double ), \
         ( "climbFpAngle",c_double ), \
         ( "climbAngleVRange",c_double ), \
         ( "climbAngleHRange",c_double ), \
         ( "climbRateGain",c_double ), \
         ( "maxClimbRate",c_double ), \
         ( "minClimbRate",c_double ), \
         ( "maxCap",c_double ), \
         ( "minCap",c_double ), \
         ( "maxSpeed",c_double ), \
         ( "minSpeed",c_double ), \
         ( "yawForward",c_bool ) \
        ]

double3 = c_double*3

class Guidance():
     def __init__(self,param):
          self.lib = CDLL('/home/josuehfa/System/icarous/Modules/lib/libGuidance.so')
          self.lib.InitGuidance.argtypes = [POINTER(GuidanceParam)]
          self.lib.InitGuidance.restype  = c_void_p
          self.lib.guidSetParams.argtypes = [c_void_p,POINTER(GuidanceParam)]
          self.lib.guidSetAircraftState.argtypes = [c_void_p,c_double*3,c_double*3]
          self.lib.guidInputFlightplanData.argtypes = [c_void_p,c_char_p,Waypoint*50,c_int,c_double,c_bool,c_double]
          self.lib.RunGuidance.argtypes = [c_void_p, c_double]
          self.lib.guidInputVelocityCmd.argtypes = [c_void_p, c_double*3]
          self.lib.guidGetOutput.argtypes = [c_void_p,POINTER(GuidanceOutput)]
          self.lib.SetGuidanceMode.argtypes = [c_void_p,c_int,c_char_p,c_int,c_bool]
          self.lib.ChangeWaypointSpeed.argtypes = [c_void_p,c_char_p,c_int,c_double,c_bool]
          self.lib.ChangeWaypointAlt.argtypes = [c_void_p,c_char_p,c_int,c_double,c_bool]
          self.lib.ChangeWaypointETA.argtypes = [c_void_p,c_char_p,c_int,c_double,c_bool]
          self.lib.guidGetWaypoint.argtypes = [c_void_p,c_char_p,c_int,POINTER(Waypoint)]
          self.lib.guidGetWaypoint.restype = c_int
          self.obj = self.lib.InitGuidance(byref(param))

     def SetGuidanceParams(self,param):
          self.lib.guidSetParams(self.obj,byref(param))

     def SetAircraftState(self,pos,vel):
          cpos = double3(*pos)
          cvel = double3(*vel)
          self.lib.guidSetAircraftState(self.obj,cpos,cvel)

     def GetKinematicPlan(self,planID):
          id = 0
          totalWP = 1
          fp = []
          while id < totalWP:
               wp = Waypoint()
               totalWP = self.lib.guidGetWaypoint(self.obj,c_char_p(planID.encode('utf-8')),c_int(id),byref(wp))
               id += 1
               fp.append(wp)
          
          return fp

     def InputFlightplanData(self,planID,fp,repair=False,repairTurnRate=0):
          n = len(fp)
          wparray = Waypoint*50
          wpts = wparray()
          for i,wp in enumerate(fp):
               wpts[i] = wp
          self.lib.guidInputFlightplanData(self.obj,c_char_p(planID.encode('utf-8')),
                                          wpts,c_int(n),c_double(0),c_bool(repair),c_double(repairTurnRate))

     def RunGuidance(self,time):
          self.lib.RunGuidance(self.obj,c_double(time))

     def InputVelocityCmd(self,velcmd):
          cVelCmd = double3(*velcmd)
          self.lib.guidInputVelocityCmd(self.obj,cVelCmd)

     def GetOutput(self):
          output = GuidanceOutput()
          self.lib.guidGetOutput(self.obj,byref(output))
          return output

     def SetGuidanceMode(self,mode,planID,nextWP,eta=False):
          self.lib.SetGuidanceMode(self.obj,c_int(mode),c_char_p(planID.encode('utf-8')),c_int(nextWP),c_bool(eta))

     def ChangeWaypointSpeed(self,planID,wpID,val,updateAll):
          self.lib.ChangeWaypointSpeed(self.obj,c_char_p(planID.encode('utf-8')),c_int(wpID),c_double(val),c_bool(updateAll))

     def ChangeWaypointAlt(self,planID,wpID,val,updateAll):
          self.lib.ChangeWaypointAlt(self.obj,c_char_p(planID.encode('utf-8')),c_int(wpID),c_double(val),c_bool(updateAll))

     def ChangeWaypointETA(self,planID,wpID,val,updateAll):
          self.lib.ChangeWaypointETA(self.obj,c_char_p(planID.encode('utf-8')),
                                c_int(wpID),c_double(val),c_bool(updateAll))
