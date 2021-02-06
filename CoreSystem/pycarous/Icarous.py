from ctypes import byref
import numpy as np
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from OptimalClass import *
from MapGenClass import *
from PlotlyClass import *
from ScenarioClass import *
from Interfaces import CommandTypes, GeofenceConflict
from Cognition import Cognition, CognitionParams
from Guidance import Guidance, GuidanceParam, GuidanceMode
from Geofence import GeofenceMonitor
from Trajectory import Trajectory,DubinsParams
from TrafficMonitor import TrafficMonitor
from Merger import MAX_NODES, Merger, LogData, MergerData
from ichelper import (ConvertTrkGsVsToVned,
                      ConvertVnedToTrkGsVs,
                      distance,
                      Getfence,
                      gps_offset,
                      GetFlightplan,
                      ConvertVnedToTrkGsVs,
                      ComputeHeading,
                      LoadIcarousParams,
                      ReadFlightplanFile,
                      ConstructWaypointsFromList)
import time
from IcarousInterface import IcarousInterface

class Icarous(IcarousInterface):
    """
    Interface to launch and control an instance of pycarous
    (core ICAROUS modules called from python)
    """
    def __init__(self, home_pos, callsign="SPEEDBIRD", vehicleID=0, verbose=1,
                 logRateHz=5, fasttime=True, simtype="UAS_ROTOR",
                 monitor="DAIDALUS", daaConfig="data/DaidalusQuadConfig.txt"):
        """
        Initialize pycarous
        :param fasttime: when fasttime is True, simulation will run in fasttime
        :param simtype: string defining vehicle model: UAS_ROTOR, UAM_VTOL, ...
        :param monitor: string defining DAA module: DAIDALUS, ACAS, ...
        :param daaConfig: DAA module configuration file
        Other parameters are defined in parent class, see IcarousInterface.py
        """
        super().__init__(home_pos, callsign, vehicleID, verbose)

        self.simType = "pycarous"
        self.fasttime = fasttime
        if self.fasttime:
            self.currTime = 0
        else:
            self.currTime = time.time()

        # Initialize vehicle sim
        if simtype == "UAM_VTOL":
            from vehiclesim import UamVtolSim
            self.ownship = UamVtolSim(self.vehicleID, home_pos)
        elif simtype == "UAS_ROTOR":
            from vehiclesim import QuadSim
            self.ownship = QuadSim(self.vehicleID, home_pos)
        elif simtype == "UAM_SPQ":
            from vehiclesim import SixPassengerQuadSim
            self.ownship = SixPassengerQuadSim(self.vehicleID, home_pos)

        # Initialize ICAROUS apps
        self.Cog = Cognition(self.callsign)
        self.Guidance = Guidance(GuidanceParam())
        self.Geofence = GeofenceMonitor([3, 2, 2, 20, 20])
        self.Trajectory = Trajectory(self.callsign)
        self.tfMonitor = TrafficMonitor(self.callsign, daaConfig, False, monitor)
        self.Merger = Merger(self.callsign, self.vehicleID)

        # Merger data
        self.arrTime = None
        self.logLatency = 0
        self.prevLogUpdate = 0
        self.mergerLog = LogData()
        self.localMergeFixes = []

        # Fligh plan data
        self.flightplan1 = []
        self.flightplan2 = []
        self.etaFP1      = False
        self.etaFP2      = False
        self.nextWP1 = 1
        self.nextWP2 = 1
        self.plans = []
        self.localPlans = []
        self.activePlan = "Plan0"
        self.numSecPlan = 0

        # Geofence data
        self.fences = []
        self.localFences= []
        self.numFences = 0

        # Other aircraft data
        self.guidanceMode = GuidanceMode.NOOP
        self.emergbreak = False
        self.daa_radius = 0
        self.turnRate = 0
        self.fphases = -1
        self.land    = False
        self.ditch   = False

        self.loopcount = 0

        #teste
        self.fp = []

    def SetPosUncertainty(self, xx, yy, zz, xy, yz, xz, coeff=0.8):
        self.ownship.SetPosUncertainty(xx, yy, zz, xy, yz, xz, coeff)

    def InputTraffic(self,idx,position,velocity):
        if idx is not self.vehicleID and 0 not in position:
            trkgsvs = ConvertVnedToTrkGsVs(velocity[0],velocity[1],velocity[2])
            self.tfMonitor.input_traffic(idx,position,trkgsvs,self.currTime)
            self.Trajectory.InputTrafficData(idx,position,trkgsvs,self.currTime)
            localPos = self.ConvertToLocalCoordinates(position)
            self.RecordTraffic(idx, position, velocity, localPos)

    def InputFlightplan(self,fp,eta=False,repair=False):
        
        #Modified
        self.fp = self.fp + fp
        waypoints = ConstructWaypointsFromList(self.fp,eta) 
        
        
        self.etaFP1 = eta
        self.localPlans = [self.GetLocalFlightPlan(waypoints)]
        self.Cog.InputFlightplanData("Plan0",waypoints,repair,self.turnRate)
        self.Guidance.InputFlightplanData("Plan0",waypoints,repair,self.turnRate)
        self.Trajectory.InputFlightplanData("Plan0",waypoints,repair,self.turnRate)

        if repair: 
            waypoints = self.Guidance.GetKinematicPlan("Plan0")

        #Modified
        self.plans = [waypoints]
        
        self.flightplan1 = waypoints 
        self.flightplan2 = []

    def InputFlightplanFromFile(self,filename,scenarioTime=0,eta=False,repair=False):
        fp = GetFlightplan(filename,self.defaultWPSpeed,eta) 
        self.InputFlightplan(fp,eta,repair)

    def InputGeofence(self,filename):
        self.fenceList = Getfence(filename)
        for fence in self.fenceList:
            self.Geofence.InputData(fence)
            self.Trajectory.InputGeofenceData(fence)
            self.numFences += 1

            localFence = []
            gf = []
            for vertex in fence['Vertices']:
                localFence.append(self.ConvertToLocalCoordinates([*vertex,0]))
                gf.append([*vertex,0])

            self.localFences.append(localFence)
            self.fences.append(gf)

    def InputMergeFixes(self,filename):
        wp, ind, _, _, _ = ReadFlightplanFile(filename)
        self.localMergeFixes = list(map(self.ConvertToLocalCoordinates, wp))
        self.mergeFixes = wp
        self.Merger.SetIntersectionData(list(zip(ind,wp)))

    def SetGeofenceParams(self,params):
        gfparams = [params['LOOKAHEAD'],
                    params['HTHRESHOLD'],
                    params['VTHRESHOLD'],
                    params['HSTEPBACK'],
                    params['VSTEPBACK']]
        self.Geofence.SetParameters(gfparams)

    def SetTrajectoryParams(self,params):

        # RRT Params
        dubinsParams = DubinsParams()
        dubinsParams.turnRate = params['TURN_RATE']
        dubinsParams.zSections = params['ALT_DH'] 
        dubinsParams.vertexBuffer = params['OBSBUFFER']
        dubinsParams.maxGS = params['MAX_GS'] * 0.5
        dubinsParams.minGS = params['MIN_GS'] * 0.5
        dubinsParams.maxVS = params['MAX_VS'] * 0.00508
        dubinsParams.minVS = params['MIN_VS'] * 0.00508
        dubinsParams.hAccel = params['HORIZONTAL_ACCL'] 
        dubinsParams.vAccel = params['VERTICAL_ACCL'] 
        dubinsParams.hDaccel = params['HORIZONTAL_ACCL']*-0.8
        dubinsParams.vDaccel = params['VERTICAL_ACCL']*-1
        dubinsParams.climbgs = params['CLMB_SPEED']
        dubinsParams.maxH = params['MAX_ALT']*0.3 
        dubinsParams.wellClearDistH = params['DET_1_WCV_DTHR']/3
        dubinsParams.wellClearDistV = params['DET_1_WCV_ZTHR']/3

        self.Trajectory.UpdateDubinsPlannerParams(dubinsParams)

    def SetGuidanceParams(self,params):
        guidParams = GuidanceParam()
        guidParams.defaultWpSpeed = params['DEF_WP_SPEED'] 
        guidParams.captureRadiusScaling = params['CAP_R_SCALING']
        guidParams.guidanceRadiusScaling = params['GUID_R_SCALING']
        guidParams.xtrkDev = params['XTRKDEV']
        guidParams.climbFpAngle = params['CLIMB_ANGLE']
        guidParams.climbAngleVRange = params['CLIMB_ANGLE_VR']
        guidParams.climbAngleHRange = params['CLIMB_ANGLE_HR']
        guidParams.climbRateGain = params['CLIMB_RATE_GAIN']
        guidParams.maxClimbRate = params['MAX_VS'] * 0.00508
        guidParams.minClimbRate = params['MIN_VS'] * 0.00508
        guidParams.maxCap = params['MAX_CAP']
        guidParams.minCap = params['MIN_CAP']
        guidParams.maxSpeed = params['MAX_GS'] * 0.5
        guidParams.minSpeed = params['MIN_GS'] * 0.5
        guidParams.yawForward = True if params['YAW_FORWARD'] == 1 else False
        self.defaultWPSpeed = guidParams.defaultWpSpeed
        self.Guidance.SetGuidanceParams(guidParams)

    def SetCognitionParams(self,params):
        cogParams = CognitionParams()
        cogParams.resolutionType = int(params['RES_TYPE'])
        cogParams.allowedXTrackDeviation = params['XTRKDEV']
        cogParams.DTHR = params['DET_1_WCV_DTHR']/3
        cogParams.ZTHR = params['DET_1_WCV_ZTHR']/3
        cogParams.lookaheadTime = params['LOOKAHEAD_TIME']
        cogParams.persistence_time = params['PERSIST_TIME']
        cogParams.return2nextWP = int(params['RETURN_WP'])
        self.Cog.InputParameters(cogParams)

    def SetTrafficParams(self,params):
        paramString = "" \
        +"lookahead_time="+str(params['LOOKAHEAD_TIME'])+ "[s];"\
        +"left_hdir="+str(params['LEFT_TRK'])+ "[deg];"\
        +"right_hdir="+str(params['RIGHT_TRK'])+ "[deg];"\
        +"min_hs="+str(params['MIN_GS'])+ "[knot];"\
        +"max_hs="+str(params['MAX_GS'])+ "[knot];"\
        +"min_vs="+str(params['MIN_VS'])+ "[fpm];"\
        +"max_vs="+str(params['MAX_VS'])+ "[fpm];"\
        +"min_alt="+str(params['MIN_ALT'])+ "[ft];"\
        +"max_alt="+str(params['MAX_ALT'])+ "[ft];"\
        +"step_hdir="+str(params['TRK_STEP'])+ "[deg];"\
        +"step_hs="+str(params['GS_STEP'])+ "[knot];"\
        +"step_vs="+str(params['VS_STEP'])+ "[fpm];"\
        +"step_alt="+str(params['ALT_STEP'])+ "[ft];"\
        +"horizontal_accel="+str(params['HORIZONTAL_ACCL'])+ "[m/s^2];"\
        +"vertical_accel="+str(params['VERTICAL_ACCL'])+ "[m/s^2];"\
        +"turn_rate="+str(params['TURN_RATE'])+ "[deg/s];"\
        +"bank_angle="+str(params['BANK_ANGLE'])+ "[deg];"\
        +"vertical_rate="+str(params['VERTICAL_RATE'])+ "[fpm];"\
        +"recovery_stability_time="+str(params['RECOV_STAB_TIME'])+ "[s];"\
        +"min_horizontal_recovery="+str(params['MIN_HORIZ_RECOV'])+ "[ft];"\
        +"min_vertical_recovery="+str(params['MIN_VERT_RECOV'])+ "[ft];"\
        +"recovery_hdir="+( "true;" if params['RECOVERY_TRK'] == 1 else "false;" )\
        +"recovery_hs="+( "true;" if params['RECOVERY_GS'] == 1 else "false;" )\
        +"recovery_vs="+( "true;" if params['RECOVERY_VS'] == 1 else "false;" )\
        +"recovery_alt="+( "true;" if params['RECOVERY_ALT'] == 1 else "false;" )\
        +"ca_bands="+( "true;" if params['CA_BANDS'] == 1 else "false;" )\
        +"ca_factor="+str(params['CA_FACTOR'])+ ";"\
        +"horizontal_nmac="+str(params['HORIZONTAL_NMAC'])+ "[ft];"\
        +"vertical_nmac="+str(params['VERTICAL_NMAC'])+ "[ft];"\
        +"conflict_crit="+( "true;" if params['CONFLICT_CRIT'] == 1 else "false;" )\
        +"recovery_crit="+( "true;" if params['RECOVERY_CRIT'] == 1 else "false;" )\
        +"contour_thr="+str(params['CONTOUR_THR'])+ "[deg];"\
        +"alerters = default;"\
        +"default_alert_1_detector=det_1;"\
        +"default_alert_1_region=NEAR;"\
        +"default_alert_1_alerting_time="+str(params['AL_1_ALERT_T'])+ "[s];"\
        +"default_alert_1_early_alerting_time="+str(params['AL_1_E_ALERT_T'])+ "[s];"\
        +"default_alert_1_spread_alt="+str(params['AL_1_SPREAD_ALT'])+ "[ft];"\
        +"default_alert_1_spread_hs="+str(params['AL_1_SPREAD_GS'])+ "[knot];"\
        +"default_alert_1_spread_hdir="+str(params['AL_1_SPREAD_TRK'])+ "[deg];"\
        +"default_alert_1_spread_vs="+str(params['AL_1_SPREAD_VS'])+ "[fpm];"\
        +"default_det_1_WCV_DTHR="+str(params['DET_1_WCV_DTHR'])+ "[ft];"\
        +"default_det_1_WCV_TCOA="+str(params['DET_1_WCV_TCOA'])+ "[s];"\
        +"default_det_1_WCV_TTHR="+str(params['DET_1_WCV_TTHR'])+ "[s];"\
        +"default_det_1_WCV_ZTHR="+str(params['DET_1_WCV_ZTHR'])+ "[ft];"\
        +"default_load_core_detection_det_1="+"gov.nasa.larcfm.ACCoRD.WCV_TAUMOD;"\
        +"default_alert_2_detector=det_2;"\
        +"default_alert_2_region=NEAR;"\
        +"default_alert_2_alerting_time= 0.0 [s];"\
        +"default_alert_2_early_alerting_time= 0.0 [s];"\
        +"default_alert_2_spread_alt="+str(params['AL_1_SPREAD_ALT'])+ "[ft];"\
        +"default_alert_2_spread_hs="+str(params['AL_1_SPREAD_GS'])+ "[knot];"\
        +"default_alert_2_spread_hdir="+str(params['AL_1_SPREAD_TRK'])+ "[deg];"\
        +"default_alert_2_spread_vs="+str(params['AL_1_SPREAD_VS'])+ "[fpm];"\
        +"default_det_2_D="+str(params['DET_1_WCV_DTHR'])+ "[ft];"\
        +"default_det_2_H="+str(params['DET_1_WCV_ZTHR'])+ "[ft];"\
        +"default_load_core_detection_det_2="+"gov.nasa.larcfm.ACCoRD.CDCylinder;"
        daa_log = True if params['LOGDAADATA'] == 1 else False
        self.tfMonitor.SetParameters(paramString,daa_log)
        self.daa_radius = params['DET_1_WCV_DTHR']/3
        self.turnRate = params['TURN_RATE']

    def SetMergerParams(self,params):
        self.Merger.SetVehicleConstraints(params["MIN_GS"]*0.5,
                                          params["MAX_GS"]*0.5,
                                          params["MAX_TURN_RADIUS"])
        self.Merger.SetFixParams(params["MIN_SEP_TIME"],
                                 params["COORD_ZONE"],
                                 params["SCHEDULE_ZONE"],
                                 params["ENTRY_RADIUS"],
                                 params["CORRIDOR_WIDTH"])
        
    def SetParameters(self,params):
        self.params = params
        self.SetGuidanceParams(params)
        self.SetGeofenceParams(params)
        self.SetTrajectoryParams(params)
        self.SetCognitionParams(params)
        self.SetTrafficParams(params)
        self.SetMergerParams(params)

    def RunOwnship(self):
        trkgsvs_command = ConvertVnedToTrkGsVs(*self.controlInput)
        self.ownship.InputCommand(*trkgsvs_command)
        self.ownship.Run(windFrom=self.windFrom,windSpeed=self.windSpeed)

        opos = self.ownship.GetOutputPositionNED()
        ovel = self.ownship.GetOutputVelocityNED()

        self.localPos = opos

        (ogx, ogy) = gps_offset(self.home_pos[0], self.home_pos[1], opos[1], opos[0])
       
        self.position = [ogx, ogy, opos[2]]
        self.velocity = ovel
        self.trkgsvs = ConvertVnedToTrkGsVs(ovel[0],ovel[1],ovel[2])
        self.RecordOwnship()

    def RunCognition(self):
        self.Cog.InputVehicleState(self.position,self.trkgsvs,self.trkgsvs[0])
        
        nextWP1 = self.nextWP1
        if self.nextWP1 >= len(self.flightplan1):
            nextWP1 = len(self.flightplan1) - 1
        dist = distance(self.position[0],self.position[1],self.flightplan1[nextWP1].latitude,self.flightplan1[nextWP1].longitude)

        if self.verbose > 1:
            if self.activePlan == 'Plan0':
                print("%s: %s, Distance to wp %d: %f" %(self.callsign,self.activePlan,self.nextWP1,dist))
            else:
                print("%s: %s, Distance to wp %d: %f" %(self.callsign,self.activePlan,self.nextWP2,dist))

        self.fphase = self.Cog.RunCognition(self.currTime)
        if self.fphase == -2:
            self.land = True

        n, cmd = self.Cog.GetOutput()

        if n > 0:
            if cmd.commandType == CommandTypes.FP_CHANGE:
                self.guidanceMode = GuidanceMode.FLIGHTPLAN
                self.activePlan =  cmd.commandU.fpChange.name.decode('utf-8')
                nextWP = cmd.commandU.fpChange.wpIndex
                eta = False
                if self.activePlan == "Plan0":
                    self.flightplan2 = []
                    self.nextWP1 = nextWP
                    eta = self.etaFP1
                else:
                    self.nextWP2 = nextWP
                    eta = self.etaFP2
                self.Guidance.SetGuidanceMode(self.guidanceMode,self.activePlan,nextWP,eta)
                if self.verbose > 0:
                    print(self.callsign, ": active plan = ",self.activePlan)
            elif cmd.commandType == CommandTypes.P2P:
                self.guidanceMode = GuidanceMode.POINT2POINT
                point = cmd.commandU.p2pCommand.point
                speed = cmd.commandU.p2pCommand.speed
                fp = [(*self.position,0),(*point,speed)]
                waypoints = ConstructWaypointsFromList(fp)
                self.Guidance.InputFlightplanData("P2P",waypoints)
                self.activePlan = "P2P"
                self.nextWP2 = 1
                self.Guidance.SetGuidanceMode(self.guidanceMode,self.activePlan,1,False)
                if self.verbose > 0:
                    print(self.callsign, ": active plan = ",self.activePlan)
            elif cmd.commandType == CommandTypes.VELOCITY:
                self.guidanceMode = GuidanceMode.VECTOR
                vn = cmd.commandU.velocityCommand.vn
                ve = cmd.commandU.velocityCommand.ve
                vu = -cmd.commandU.velocityCommand.vu
                trkGsVs = ConvertVnedToTrkGsVs(vn,ve,vu)
                self.Guidance.InputVelocityCmd(trkGsVs)
                self.Guidance.SetGuidanceMode(self.guidanceMode,"",0)
            elif cmd.commandType == CommandTypes.TAKEOFF:
                self.guidanceMode = GuidanceMode.TAKEOFF
            elif cmd.commandType == CommandTypes.SPEED_CHANGE:
                self.etaFP1 = False
                speedChange =  cmd.commandU.speedChange.speed
                planID= cmd.commandU.speedChange.name.decode('utf-8')
                nextWP = self.nextWP1 if planID == "Plan0" else self.nextWP2
                self.Guidance.ChangeWaypointSpeed(planID,nextWP,speedChange,False)
            elif cmd.commandType == CommandTypes.ALT_CHANGE:
                altChange = cmd.commandU.altChange.altitude
                planID= cmd.commandU.altChange.name.decode('utf-8')
                nextWP = self.nextWP1 if planID == "Plan0" else self.nextWP2
                self.Guidance.ChangeWaypointAlt(planID,nextWP,altChange,cmd.commandU.altChange.hold)
            elif cmd.commandType == CommandTypes.STATUS_MESSAGE:
                if self.verbose > 0:
                    message = cmd.commandU.statusMessage.buffer
                    print(self.callsign,":",message.decode('utf-8'))
            elif cmd.commandType == CommandTypes.FP_REQUEST:
                self.flightplan2 = []
                self.numSecPlan += 1
                startPos = cmd.commandU.fpRequest.fromPosition
                stopPos = cmd.commandU.fpRequest.toPosition
                startVel = cmd.commandU.fpRequest.startVelocity
                endVel = cmd.commandU.fpRequest.endVelocity
                planID = cmd.commandU.fpRequest.name.decode('utf-8')
                # Find the new path
                numWP = self.Trajectory.FindPath(
                        planID,
                        startPos,
                        stopPos,
                        startVel,
                        endVel)

                if numWP > 0:
                    if self.verbose > 0:
                        print("%s : At %f s, Computed flightplan %s with %d waypoints" % (self.callsign,self.currTime,planID,numWP))
                    # offset the new path with current time (because new paths start with t=0) 
                    self.Trajectory.SetPlanOffset(planID,0,self.currTime)

                    # Combine new plan with rest of old plan
                    self.Trajectory.CombinePlan(planID,"Plan0",-1)
                    self.flightplan2 = self.Trajectory.GetFlightPlan(planID)
                    self.Cog.InputFlightplanData(planID,self.flightplan2)
                    self.Guidance.InputFlightplanData(planID,self.flightplan2)

                    # Set guidance mode with new flightplan and wp 0 to offset plan times
                    self.Guidance.SetGuidanceMode(GuidanceMode.FLIGHTPLAN,planID,0,False)

                    self.plans.append(self.flightplan2)
                    self.localPlans.append(self.GetLocalFlightPlan(self.flightplan2))
                else:
                    if self.verbose > 0:
                        print(self.callsign,"Error finding path")

    def RunGuidance(self):
        if self.guidanceMode is GuidanceMode.NOOP:
            return
        if self.guidanceMode is GuidanceMode.TAKEOFF:
            self.Cog.InputReachedWaypoint("Takeoff",0); 
            return

        self.Guidance.SetAircraftState(self.position,self.trkgsvs)

        self.Guidance.RunGuidance(self.currTime)

        guidOutput = self.Guidance.GetOutput()
        self.controlInput = ConvertTrkGsVsToVned(guidOutput.velCmd[0],
                                                 guidOutput.velCmd[1],
                                                 guidOutput.velCmd[2])

        nextWP = guidOutput.nextWP
        if self.guidanceMode is not GuidanceMode.VECTOR:
            if self.activePlan == "Plan0":
                if self.nextWP1 < nextWP:
                    self.Cog.InputReachedWaypoint("Plan0",nextWP-1)
                    self.nextWP1 = nextWP
                    if self.verbose > 0 and self.nextWP1 < len(self.flightplan1):
                        print("%s : Proceeding to waypoint %d on %s" % (self.callsign,nextWP,self.activePlan))
            else:
                if self.nextWP2 < nextWP:
                    self.nextWP2 = nextWP
                    self.Cog.InputReachedWaypoint(self.activePlan,nextWP-1)
                    if self.verbose > 0 and self.nextWP2 < len(self.flightplan2):
                        print("%s : Proceeding to waypoint %d on %s" % (self.callsign,nextWP,self.activePlan))
        else:
            pass
         
    def RunGeofenceMonitor(self):
        gfConflictData = GeofenceConflict()
        
        self.Geofence.CheckViolation(self.position,*self.trkgsvs)

        gfConflictData.numConflicts = self.Geofence.GetNumConflicts()
        gfConflictData.numFences = self.numFences
        for i in range(gfConflictData.numConflicts):
            (ids,conflict,violation,recPos,time,ftype) = self.Geofence.GetConflict(i)
            gfConflictData.conflictFenceIDs[i] = ids
            gfConflictData.conflictTypes[i] = ftype
            gfConflictData.recoveryPosition = recPos
            gfConflictData.timeToViolation[i] = time

        self.Cog.InputGeofenceConflictData(gfConflictData)

    def RunTrafficMonitor(self):
        self.tfMonitor.monitor_traffic(self.position,self.trkgsvs,self.currTime)
       
        trkband = self.tfMonitor.GetTrackBands()
        gsband = self.tfMonitor.GetGSBands()
        altband = self.tfMonitor.GetAltBands()
        vsband = self.tfMonitor.GetVSBands()

        trkband.time = self.currTime
        gsband.time = self.currTime
        altband.time = self.currTime
        vsband.time = self.currTime

        self.Cog.InputBands(trkband,gsband,altband,vsband) 

        self.trkband = trkband
        self.gsband = gsband
        self.altband = altband
        self.vsband = vsband

    def RunTrajectoryMonitor(self):
        nextWP = self.nextWP1 if self.activePlan == "Plan0" else self.nextWP2
        planID = "Plan+" if self.activePlan != "Plan0" else "Plan0"
        tjMonData = self.Trajectory.MonitorTrajectory(self.currTime,planID,self.position,self.trkgsvs,self.nextWP1,nextWP)
        self.Cog.InputTrajectoryMonitorData(tjMonData)
        self.planoffsets = [tjMonData.offsets1[0],tjMonData.offsets1[1],tjMonData.offsets1[2],\
                            tjMonData.offsets2[0],tjMonData.offsets2[1],tjMonData.offsets2[2]]

    def RunMerger(self):
        self.Merger.SetAircraftState(self.position,self.velocity)
        if self.currTime - self.prevLogUpdate > self.logLatency:
            self.Merger.SetMergerLog(self.mergerLog)
            self.prevLogUpdate = self.currTime

        mergingActive = self.Merger.Run(self.currTime)
        self.Cog.InputMergeStatus(mergingActive)
        outAvail, arrTime =  self.Merger.GetArrivalTimes()
        if outAvail:
            if arrTime.intersectionID > 0:
                self.arrTime = arrTime
        if mergingActive == 3:
            velCmd = self.Merger.GetVelocityCmd()
            speedChange = velCmd[1]
            nextWP = self.nextWP1
            self.Guidance.ChangeWaypointSpeed("Plan0",nextWP,speedChange,False)

    def InputMergeLogs(self,logs,delay):
        self.mergerLog = logs
        self.logLatency = delay

    def StartMission(self):
        self.Cog.InputStartMission(0, 0)
        self.missionStarted = True

    def CheckMissionComplete(self):
        if self.land:
            self.missionComplete = True
        return self.missionComplete

    def Run(self):
        time_now = time.time()
        if not self.fasttime:
            if time_now - self.currTime >= 0.05:
                self.ownship.dt = time_now - self.currTime
                self.currTime = time_now
            else:
                return False
        else:
            self.currTime += 0.05

        if self.CheckMissionComplete():
            return True

        self.loopcount += 1

        time_ship_in = time.time()
        self.RunOwnship()
        time_ship_out = time.time()

        if not self.missionStarted:
            return True

        time_cog_in = time.time()
        self.RunCognition()
        time_cog_out = time.time()
        
        time_traff_in = time_traff_out = 0.0
        if self.loopcount%1 == 0:
            time_traff_in = time.time()
            self.RunTrafficMonitor()
            time_traff_out = time.time()

        time_geof_in = time.time()
        self.RunGeofenceMonitor()
        time_geof_out = time.time()

        time_guid_in = time.time()
        self.RunGuidance()
        time_guid_out = time.time()

        self.RunTrajectoryMonitor()

        self.RunMerger()

        #print("cog     %f" % (time_cog_out - time_cog_in))
        #print("geofence %f" % (time_geof_out - time_geof_in))
        #print("ownship %f" % (time_ship_out - time_ship_in))
        #print("traffic %f" % (time_traff_out - time_traff_in))

        return True

    def Terminate(self):
        # No special shut down action required when using pycarous
        pass


def VisualizeSimData(icList,allplans=False,showtraffic=True,xmin=-100,ymin=-100,xmax=100,ymax=100,playbkspeed=1,interval=30,record=False,filename=""):
    '''
    ic: icarous object
    allplans: True - plot all computed plans, False - plot only the mission plan
    xmin,ymin : plot axis min values
    xmax,ymax : plot axis max values
    interval  : Interval between frames
    '''
    if record:
        import matplotlib; matplotlib.use('Agg')
    from Animation import AgentAnimation
    anim= AgentAnimation(xmin,ymin, xmax,ymax,playbkspeed,interval,record,filename)

    vehicleSize1 = np.abs(xmax - xmin)/10000
    vehicleSize2 = np.abs(ymax - ymin)/10000
    vehicleSize  = np.max([vehicleSize1,vehicleSize2])
    for j,ic in enumerate(icList):
        anim.AddAgent('ownship'+str(j),vehicleSize,'r',ic.ownshipLog,show_circle=True,circle_rad=ic.daa_radius)
        for i,pln in enumerate(ic.localPlans):
            if i == 0:
                anim.AddPath(np.array(pln),'k--')

            if i > 0 and allplans:
                anim.AddPath(np.array(pln),'k--')
        tfids = ic.trafficLog.keys()
        if showtraffic:
            for key in tfids:
                anim.AddAgent('traffic'+str(key),vehicleSize,'b',ic.trafficLog[key])
        for fence in ic.localFences:
            fence.append(fence[0])
            anim.AddFence(np.array(fence),'c-.')
    for fix in icList[0].localMergeFixes:
        anim.AddZone(fix[::-1][1:3],icList[0].params['COORD_ZONE'],'r')
        anim.AddZone(fix[::-1][1:3],icList[0].params['SCHEDULE_ZONE'],'b')
        anim.AddZone(fix[::-1][1:3],icList[0].params['ENTRY_RADIUS'],'g')

    anim.run()


def VisualizeSimDataOptimal(icList,scenario_time, scenario,allplans=True,showtraffic=True,xmin=-100,ymin=-100,xmax=100,ymax=100,playbkspeed=1,interval=30,record=False,filename=""):
    '''
    ic: icarous object
    allplans: True - plot all computed plans, False - plot only the mission plan
    xmin,ymin : plot axis min values
    xmax,ymax : plot axis max values
    interval  : Interval between frames
    '''
    if record:
        import matplotlib; matplotlib.use('Agg')
    from Animation import AgentAnimation
    anim = AgentAnimation(xmin,ymin, xmax,ymax,icList[0].ownshipLog,scenario, scenario_time,playbkspeed,interval,record,filename)

    #scenario = ScenarioClass(scenario)

    vehicleSize1 = np.abs(xmax - xmin)/3000
    vehicleSize2 = np.abs(ymax - ymin)/3000
    vehicleSize  = np.max([vehicleSize1,vehicleSize2])
    #Tentativa de adicionar o mapa de custos
    anim.AddCostMap(scenario.mapgen,xmin,xmax,ymin,ymax)
    anim.AddObstacles(scenario)
    anim.AddZoom()
    anim.AddWaypoint()
    anim.AddStartGoal()
    for j,ic in enumerate(icList):
        anim.AddAgent('ownship'+str(j),vehicleSize,'r',ic.ownshipLog,show_circle=True,circle_rad=ic.daa_radius)
        #for i,pln in enumerate(ic.localPlans):
        #    pln_aux = []
        #    for k, vec in enumerate(pln):
        #        vec[1] = ic.localCoords[k][0]
        #        vec[2] = ic.localCoords[k][1]
        #        pln_aux.append(vec)

        #    if i == 0:
        #        anim.AddPath(np.array(pln_aux),'k--')
                
        #    if i > 0 and allplans:
        #        anim.AddPath(np.array(pln_aux),'k--')
        
        tfids = ic.trafficLog.keys()
        if showtraffic:
            for key in tfids:
                anim.AddAgent('traffic'+str(key),vehicleSize,'b',ic.trafficLog[key])
        for fence in ic.localFences:
            fence.append(fence[0])
            anim.AddFence(np.array(fence),'c-.')
    for fix in icList[0].localMergeFixes:
        anim.AddZone(fix[::-1][1:3],icList[0].params['COORD_ZONE'],'r')
        anim.AddZone(fix[::-1][1:3],icList[0].params['SCHEDULE_ZONE'],'b')
        anim.AddZone(fix[::-1][1:3],icList[0].params['ENTRY_RADIUS'],'g')

    anim.run()
