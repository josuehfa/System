import sys, os
import numpy as np
import time
from euclid import *
from vehiclesim import UamVtolSim
from ichelper import distance, ConvertVnedToTrkGsVs
from Merger import LogData, MergerData, MAX_NODES
from Interfaces import V2Vdata
from communicationmodels import channelmodels as cm
from communicationmodels import get_propagation_model, get_reception_model
from communicationmodels import get_transmitter, get_receiver
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from OptimalClass import *
from MapGenClass import *
from PlotlyClass import *
from ScenarioClass import *

class SimEnvironment:
    """ Class to manage pycarous fast time simulations """
    def __init__(self, propagation_model="NoLoss", reception_model="Perfect",
                 propagation_params={}, reception_params={}, verbose=1,
                 fasttime=True, time_limit=None):
        """
        :param propagation_model: name of signal propagation model
            ex: "NoLoss", "FreeSpace", "TwoRayGround"
        :param reception_model: name of V2V reception model,
            ex: "Perfect", "Deterministic", "Rayleigh", "Nakagami"
        :param propagation_params: dict of keyword params for propagation model
        :param reception_params: dict of keyword params for reception model
        """
        self.verbose = verbose

        # Vehicle instances
        self.icInstances = []
        self.icStartDelay = []
        self.icTimeLimit = []
        self.tfList = []

        # Communication channel
        pm = get_propagation_model(propagation_model, propagation_params)
        reception_params["propagation_model"] = pm
        rm = get_reception_model(reception_model, reception_params)
        self.comm_channel = cm.ChannelModel(pm, rm)
        if verbose > 0:
            print("Reception model: %s" % rm.model_name)

        # Simulation environment conditions
        self.wind = [(0, 0)]
        self.commDelay = 0
        self.mergeFixFile = None
        self.home_gps = [0, 0, 0]
        self.fasttime = fasttime
        self.dT = 0.05
        self.time_limit = time_limit

        # Simulation status
        self.count = 0
        self.current_time = 0
        self.windFrom, self.windSpeed = self.wind[0]

    def AddIcarousInstance(self, ic, delay=0, time_limit=1000,
                           transmitter="GroundTruth", receiver="GroundTruth"):
        """
        Add an Icarous instance to the simulation environment
        :param ic: An Icarous instance
        :param delay: Time to delay before starting mission (s)
        :param time_limit: Time limit to fly before shutting vehicle down (s)
        :param transmitter: A Transmitter to send V2V position data,
        ex: "ADS-B" or "GroundTruth"
        :param receiver: A Receiver to get V2V position data,
        ex: "ADS-B" or "GroundTruth"
        """
        self.icInstances.append(ic)
        self.icStartDelay.append(delay)
        self.icTimeLimit.append(time_limit)

        # Create a transmitter and receiver for V2V communications
        ic.transmitter = get_transmitter(transmitter, ic.vehicleID, self.comm_channel)
        ic.receiver = get_receiver(receiver, ic.vehicleID, self.comm_channel)

        # Set simulation home position
        if self.home_gps == [0, 0, 0]:
            self.home_gps = ic.home_pos

    def AddTraffic(self, idx, home, rng, brng, alt, speed, heading, crate,
                   transmitter="GroundTruth"):
        """
        Add a simulated virtual traffic vehicle to the simulation environment
        :param idx: traffic vehicle id
        :param home: home position (lat [deg], lon [deg], alt [m])
        :param rng: starting distance from home position [m]
        :param brng: starting bearing from home position [deg], 0 is North
        :param alt: starting altitude [m]
        :param speed: traffic speed [m/s]
        :param heading: traffic heading [deg], 0 is North
        :param crate: traffic climbrate [m/s]
        :param transmitter: A Transmitter to send V2V position data,
        ex: "ADS-B" or "GroundTruth"
        """
        tx = rng*np.sin(brng*np.pi/180)
        ty = rng*np.cos(brng*np.pi/180)
        tz = alt
        tvx = speed*np.sin(heading*np.pi/180)
        tvy = speed*np.cos(heading*np.pi/180)
        tvz = crate
        traffic = UamVtolSim(idx, home, tx, ty, tz, tvx, tvy, tvz)
        traffic.InputCommand(heading, speed, crate)
        self.tfList.append(traffic)

        # Create a transmitter for V2V communications
        traffic.transmitter = get_transmitter(transmitter, idx, self.comm_channel)

        return traffic

    def RunSimulatedTraffic(self, dT=None):
        """ Update all simulated traffic vehicles """
        for tf in self.tfList:
            tf.dt = dT or self.dT
            tf.Run(self.windFrom, self.windSpeed)
            tf_gps_pos = tf.GetOutputPositionLLA()
            tf_vel = tf.GetOutputVelocityNED()
            tfdata = V2Vdata("INTRUDER", {"id":tf.vehicleID,
                                          "pos": tf_gps_pos,
                                          "vel": tf_vel})
            tf.transmitter.transmit(self.current_time, tf_gps_pos, tfdata)

    def AddWind(self, wind):
        """
        Add a wind vector to simulation environment
        :param wind: a list of tuples representing the wind vector at each
        simulation timestep.
        ex: [(wind source [deg, 0 = North], wind speed [m/s]), ...]
        """
        self.wind = wind

    def GetWind(self):
        """ Return wind tuple for current simulation timestep """
        i = min(len(self.wind) - 1, self.count)
        return self.wind[i]

    def SetPosUncertainty(self, xx, yy, zz, xy, yz, xz, coeff=0.8):
        """
        Set position uncertainty for all existing Icarous and traffic instances
        :param xx: x position variance [m^2] (East/West)
        :param yy: y position variance [m^2] (North/South)
        :param zz: z position variance [m^2] (Up/Down)
        :param xy: xy position covariance [m^2]
        :param yz: yz position covariance [m^2]
        :param xz: xz position covariance [m^2]
        :param coeff: smoothing factor used for uncertainty (default=0.8)
        """
        for vehicle in (self.icInstances + self.tfList):
            vehicle.SetPosUncertainty(xx, yy, zz, xy, yz, xz, coeff)

    def InputMergeFixes(self, filename):
        """ Input a file to read merge fixes from """
        self.mergeFixFile = filename

    def ExchangeV2VData(self):
        """ Exchange V2V communications between the Icarous instances """

        # Transmit all V2V data
        for ic in self.icInstances:
            # Do not broadcast if running SBN
            if "SBN" in ic.apps:
                continue
            if not ic.missionStarted or ic.missionComplete:
                continue
            # broadcast position data to all other vehicles in the airspace
            tfdata = V2Vdata("INTRUDER", {"id":ic.vehicleID,
                                          "pos": ic.position,
                                          "vel": ic.velocity})
            ic.transmitter.transmit(self.current_time, ic.position, tfdata)

        # Gather logs to be exchanged for merging activity
        log = {}
        for ic in self.icInstances:
            if ic.arrTime is not None:
                intsid = ic.arrTime.intersectionID
                if intsid in log.keys():
                    log[intsid].append(ic.arrTime)
                else:
                    log[intsid] = [ic.arrTime]

        # Transmit all merger log data
        for lg in log:
            datalog = LogData()
            datalog.intersectionID = log[lg][0].intersectionID
            datalog.nodeRole = 1
            datalog.totalNodes = len(log[lg])
            mg = MergerData*MAX_NODES
            datalog.log = mg(*log[lg])
            for ic in self.icInstances:
                # Do not broadcast if running SBN
                if "SBN" in ic.apps:
                    continue
                mergelog = V2Vdata("MERGER",datalog)
                ic.transmitter.transmit(self.current_time, ic.position, mergelog)

        # Receive all V2V data
        for ic in self.icInstances:
            if not ic.missionStarted or ic.missionComplete:
                continue
            received_msgs = ic.receiver.receive(self.current_time, ic.position)
            data = [m.data for m in received_msgs]
            ic.InputV2VData(data)

        self.comm_channel.flush()

    def RunSimulation(self):
        """ Run simulation until mission complete or time limit reached """
        simComplete = False
        if self.mergeFixFile is not None:
            for ic in self.icInstances:
                ic.InputMergeFixes(self.mergeFixFile)
        if self.fasttime:
            t0 = 0
        else:
            t0 = time.time()
        self.current_time = t0

        while not simComplete:
            status = False

            # Advance time
            duration = self.current_time - t0
            if self.fasttime:
                self.count += 1
                self.current_time += self.dT
                self.RunSimulatedTraffic()
                if self.verbose > 0:
                    print("Sim Duration: %.1fs" % (duration), end="\r")
            else:
                time_now = time.time()
                if time_now - self.current_time >= self.dT:
                    dT = time_now - self.current_time
                    self.current_time = time_now
                    self.count += 1
                    self.RunSimulatedTraffic(dT=dT)
                    if self.verbose > 0:
                        print("Sim Duration: %.1fs" % (duration), end="\r")
            self.windFrom, self.windSpeed = self.GetWind()

            # Update Icarous instances
            for i, ic in enumerate(self.icInstances):
                ic.InputWind(self.windFrom, self.windSpeed)

                #Test
                #if round(self.current_time,1) == 160.0:
                #    fp = [[37.102577,-76.387807, 5.0, 1.0, [0, 0, 0], [0.0, 0, 0]]]
                #    ic.InputFlightplan(fp,False,False)

                if ic.CheckMissionComplete():
                    ic.Terminate()

                # Send mission start command
                if not ic.missionStarted and duration >= self.icStartDelay[i]:
                    ic.StartMission()
                    if self.verbose > 0:
                        print("%s : Start command sent at %f" %
                              (ic.callsign, self.current_time))

                # Run Icarous
                status |= ic.Run()

                # Check if time limit has been met
                if ic.missionStarted and not ic.missionComplete:
                    if duration >= self.icTimeLimit[i]:
                        ic.missionComplete = True
                        ic.Terminate()
                        if self.verbose > 0:
                            print("%s : Time limit reached at %f" %
                                  (ic.callsign, self.current_time))

            if self.time_limit is not None and duration >= self.time_limit:
                for ic in self.icInstances:
                    ic.missionComplete = True
                    ic.Terminate()
                    if self.verbose > 0:
                        print("Time limit reached at %f" % self.current_time)
            
            # Exchange all V2V data between vehicles in the environment
            self.ExchangeV2VData()

            simComplete = all(ic.missionComplete for ic in self.icInstances)
        self.ConvertLogsToLocalCoordinates()

    def RunSimulationOptimal(self,scenario, plans, path_x, path_y, delta_d,processing_time, planner, dimension, ims, axis, time_res, speed_aircraft):
        """ Run simulation with optimal planning until mission complete or time limit reached """
        simComplete = False
        if self.mergeFixFile is not None:
            for ic in self.icInstances:
                ic.InputMergeFixes(self.mergeFixFile)
        if self.fasttime:
            t0 = 0
        else:
            t0 = time.time()
        self.current_time = t0


        min_dist = distance(scenario.start_real[0],scenario.start_real[1],scenario.start_real[0]+scenario.lat_range*delta_d,scenario.start_real[1]+scenario.lon_range*delta_d)
        min_time = min_dist/(speed_aircraft*2)
        time_ompl = min_time
        cont = 0
        run = True

        while not simComplete:
            status = False

            # Advance time
            duration = self.current_time - t0
            if self.fasttime:
                self.count += 1
                self.current_time += self.dT
                self.RunSimulatedTraffic()
                if self.verbose > 0:
                    print("Sim Duration: %.1fs" % (duration), end="\r")
            else:
                time_now = time.time()
                if time_now - self.current_time >= self.dT:
                    dT = time_now - self.current_time
                    self.current_time = time_now
                    self.count += 1
                    self.RunSimulatedTraffic(dT=dT)
                    if self.verbose > 0:
                        print("Sim Duration: %.1fs" % (duration), end="\r")
            self.windFrom, self.windSpeed = self.GetWind()

            # Update Icarous instances
            for i, ic in enumerate(self.icInstances):
                ic.InputWind(self.windFrom, self.windSpeed)

                #Time to the last point 
                if round(self.current_time*1.05,1) == round(ic.localPlans[0][-1][0],1):
                    tried = 0
                    max_try = 1
                    while tried <= max_try:
                        plan_aux = []
                        cost_aut = []
                        if tried <= max_try:
                            #Linear algegra to return the next point in a line
                            p1 = Vector2(plans[-1].solutionSampled[0][0][0], plans[-1].solutionSampled[0][1][0])
                            p2 = Vector2(plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                            vector = p2-p1
                            vector = vector.normalize()
                            next_point = p1 + vector*delta_d
                            
                        else:
                            p1 = Vector2(plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                            p2 = Vector2(plans[-1].solutionSampled[0][0][2], plans[-1].solutionSampled[0][1][2])
                            vector = p2-p1
                            vector = vector.normalize()
                            next_point = p1 + vector*delta_d
                            
                        for idx, alg in enumerate(['RRTstar']):
                            plan_aux.append(OptimalPlanning((next_point[0],next_point[1]), scenario.goal, scenario.region, scenario.obstacle, planner, dimension, scenario.mapgen.z_time[round(cont)]))
                            result = plan_aux[idx].plan(processing_time+tried, alg, 'WeightedLengthAndClearanceCombo',delta_d)
                            if plan_aux[idx].solution != []:
                                cost_aut.append(plan_aux[idx].solution[0][3])
                            else:
                                cost_aut.append(np.inf)
                        lower_cost = cost_aut.index(min(cost_aut))

                        #Se existir solução
                        if plan_aux[lower_cost].solution != []:
                            #Se o ultimo costmap é do mesmo periodo que o atual
                            if time_ompl == time_ompl:
                                #Tentar encontrar um valor melhor que o ultimo
                                if plan_aux[lower_cost].solution[0][3] < plans[-1].solution[0][3] * 1.05 and (plan_aux[lower_cost].solution != plans[-1].solution):
                                    plans.append(plan_aux[lower_cost])
                                    PlanningStatus(scenario,plans)
                                    print('Add, Better Solution: ' +  str(time_ompl) + " : " + str(cont) + " : " + str(min_time) + " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                                    
                                    path_x.append(plans[-1].solution[0][0][0])
                                    path_y.append(plans[-1].solution[0][1][0])
                                    time_res.append(int(round(cont)))
                                    ims.append(plotResult(plans[-1],axis, scenario, path_x, path_y, round(cont)))
                                    
                                    tried = 0
                                    pnt = 1
                                    break

                                elif tried >= max_try: 
                                    plans.append(plan_aux[lower_cost])
                                    PlanningStatus(scenario,plans)
                                    print('Add, tried Solution: ' +  str(time_ompl) + " : " + str(cont) + " : " + str(min_time) + " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                                    
                                    path_x.append(plans[-1].solution[0][0][0])
                                    path_y.append(plans[-1].solution[0][1][0])
                                    time_res.append(int(round(cont)))
                                    ims.append(plotResult(plans[-1],axis, scenario, path_x, path_y, round(cont)))
                                    
                                    tried = 0
                                    pnt = 1
                                    break
                                else:
                                    print('Tried: '+ str(tried) + " - (" + str(next_point[0])+','+str(next_point[1])+")" )
                                    if tried >= max_try :
                                        pnt = pnt +1    
                                    tried = tried + 1

                            else:
                                plans.append(plan_aux[lower_cost])
                                PlanningStatus(scenario,plans)
                                print('Add, Other time: ' +  str(time_ompl) + " : " + str(cont) + " : " + str(min_time)+ " Pnt: " + str(next_point[0])+','+str(next_point[1]))
                                
                                path_x.append(plans[-1].solution[0][0][0])
                                path_y.append(plans[-1].solution[0][1][0])
                                time_res.append(int(round(cont)))
                                ims.append(plotResult(plans[-1],axis, scenario, path_x, path_y, round(cont)))

                                                
                            
                        else:
                            print('Pop Solution - Time: ' +  str(round(t)) + " : " + str(t))
                            plans.pop()
                            continue
                        

                        goal_sampled = (round(scenario.goal[0]*(scenario.mapgen.z.shape[0]-1))*delta_d, round(scenario.goal[1]*(scenario.mapgen.z.shape[0]-1))*delta_d)
                        if (plans[-1].solutionSampled[0][0][0], plans[-1].solutionSampled[0][1][0]) == goal_sampled:
                            path_x.append(plans[-1].solution[0][0][0])
                            path_y.append(plans[-1].solution[0][1][0])
                            time_res.append(int(round(cont)))
                            path_x.append(scenario.goal[0])
                            path_y.append(scenario.goal[1])
                            time_res.append(int(round(cont)))
                            run = False
                            break

                    #Tempo para reprocessar a rota
                    time_ompl = time_ompl + min_time
                    cont = cont + 0.1
                    if cont >= len(scenario.mapgen.z_time)-1:
                        cont = len(scenario.mapgen.z_time)-1

                    #fp_start = []
                    #for idx in range(len(plans[-1].solution[0][0])):
                    #    lat = plans[-1].solution[0][1][idx]*scenario.lat_range + min(scenario.lat_region)
                    #    lon = plans[-1].solution[0][0][idx]*scenario.lon_range + min(scenario.lon_region)
                        
                        #[lat, lon, alt, wp_metric, tcp, tcp_value]
                        #fp = [[37.102177, -76.387207, 5.0, 1.0, [0, 0, 0], [0.0, 0, 0]]]
                    #    if idx == 0 or idx == len(plans[-1].solution[0][0])-1:
                    #        fp_start.append([lat, lon, 0.0, speed_aircraft, [0, 0, 0], [0.0, 0, 0]])
                    #    else:
                    #        fp_start.append([lat, lon, 5.0, speed_aircraft, [0, 0, 0], [0.0, 0, 0]])

                    fp_start = []
                    #First Calculeted Waypoint
                    p1 = Vector2(plans[-1].solutionSampled[0][0][0], plans[-1].solutionSampled[0][1][0])
                    p2 = Vector2(plans[-1].solutionSampled[0][0][1], plans[-1].solutionSampled[0][1][1])
                    vector = p2-p1
                    vector = vector.normalize()
                    next_point = p1 + vector*delta_d
                    lat = next_point[1]*scenario.lat_range + min(scenario.lat_region)
                    lon = next_point[0]*scenario.lon_range + min(scenario.lon_region)
                    fp_start.append([lat, lon, 5.0, speed_aircraft, [0, 0, 0], [0.0, 0, 0]])



                    #Input flightplan
                    #Use the first plan of OMPL
                    if run:
                        ic.InputFlightplan(fp_start[:1],eta=False,repair=False)
                
                if ic.CheckMissionComplete():
                    ic.Terminate()

                # Send mission start command
                if not ic.missionStarted and duration >= self.icStartDelay[i]:
                    ic.StartMission()
                    if self.verbose > 0:
                        print("%s : Start command sent at %f" %
                              (ic.callsign, self.current_time))

                # Run Icarous
                status |= ic.Run()

                # Check if time limit has been met
                if ic.missionStarted and not ic.missionComplete:
                    if duration >= self.icTimeLimit[i]:
                        ic.missionComplete = True
                        ic.Terminate()
                        if self.verbose > 0:
                            print("%s : Time limit reached at %f" %
                                  (ic.callsign, self.current_time))

            if self.time_limit is not None and duration >= self.time_limit:
                for ic in self.icInstances:
                    ic.missionComplete = True
                    ic.Terminate()
                    if self.verbose > 0:
                        print("Time limit reached at %f" % self.current_time)
            
            # Exchange all V2V data between vehicles in the environment
            self.ExchangeV2VData()

            simComplete = all(ic.missionComplete for ic in self.icInstances)
        self.ConvertLogsToLocalCoordinates()

    def ConvertLogsToLocalCoordinates(self):    
        # Convert flightplans from all instances to a common reference frame
        to_local = self.icInstances[0].ConvertToLocalCoordinates
        for i, ic in enumerate(self.icInstances):
            ic.home_pos = self.home_gps
            posNED = list(map(to_local, ic.ownshipLog["position"]))
            ic.ownshipLog["positionNED"] = posNED
            for tfid in ic.trafficLog.keys():
                tfPosNED = list(map(to_local, ic.trafficLog[tfid]["position"]))
                ic.trafficLog[tfid]["positionNED"] = tfPosNED
            ic.localPlans = []
            ic.localFences = []
            ic.localMergeFixes = []
            for plan in ic.plans:
                wps = [[wp.latitude,wp.longitude,wp.altitude] for wp in plan]
                times = [wp.time for wp in plan]
                tcps = [[*wp.tcp] for wp in plan]
                tcpValues = [[*wp.tcpValue] for wp in plan]
                localwps = list(map(to_local,wps))
                localFP = [[val[0],*val[1],*val[2],*val[3]] for val in zip(times,localwps,tcps,tcpValues)]
                ic.localPlans.append(localFP)
            for fence in ic.fences:
                localFence = list(map(to_local, fence))
                ic.localFences.append(localFence)
            ic.localMergeFixes = list(map(to_local, ic.mergeFixes))

    def WriteLog(self):
        """ Write json logs for each icarous instance """
        for ic in self.icInstances:
            ic.WriteLog()
