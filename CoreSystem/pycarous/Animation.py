import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib import text
from math import sin,cos,atan2,pi
import numpy as np
import math


class AgentAnimation():
    def __init__(self,xmin,ymin,xmax,ymax,data,scenario,scenario_time,playbkspeed=1,interval=5,record=False,filename=""):
        self.fig = plt.figure(frameon=True)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        self.scenario = scenario
        self.current_time = 0 
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.paths = {}
        self.paths_zoom = {}
        self.waypoints = {}
        self.agents = []
        self.agents_zoom = []
        self.agentsRadius = {}
        self.agentNames = []
        self.agentLines = []
        self.data = {}
        self.interval = interval
        self.circle = {}
        self.circle_zoom = {}
        self.bands = {}
        self.bands_zoom = {}
        self.record = record
        self.status  = {}
        self.filename = filename
        self.speed = playbkspeed
        self.last_position = [0,0,0]
        self.axins = self.ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        self.region = [(xmin,ymin),(xmax,ymax)]
        x_range = xmin-xmax
        y_range = ymin-ymax
        self.axins_pos = [[xmin - (x_range)*0.65, ymin - (y_range)*0.65],[xmin - (x_range)*0.65, ymin - (y_range)*0.65 - (y_range)*0.3]]
        

        #Costmap timevector
        self.minlen = len(data['position'])
        cont = 0
        self.costtime = []
        self.pathtime = []
        minToCoord = int(round((self.minlen-1)/(len(data['localCoords'])-1)))+1
        for idx in range(0,self.minlen):
            if idx > minToCoord*(cont+1):
                cont = cont +1
            self.pathtime.append(cont)
            self.costtime.append(scenario_time[cont])
            


    def AddAgent(self,name,radius,color,data,show_circle=False,circle_rad = 10):
        #agt = plt.Circle((0.0, 0.0), radius=radius, fc=color)
        agt = self.GetTriangle(radius,(0.0,0,0),(1.0,0.0),color)
        agt_zoom = self.GetTriangle(radius,(0.0,0,0),(1.0,0.0),color)
        self.ax.add_patch(agt)
        self.axins.add_patch(agt_zoom)
        self.agents.append(agt)
        self.agents_zoom.append(agt_zoom)
        self.agentNames.append(name)
        self.data[name] = data
        self.minlen = len(data['position'])
        
        line, = plt.plot(0,0,'w')
        line_zoom, = self.axins.plot(0,0,'w')
        self.paths[name] = line
        self.paths_zoom[name] = line_zoom 
        self.agentsRadius[name] = radius
        if show_circle:
            circlePatch = plt.Circle((0, 0), radius=circle_rad, fc='y',alpha=0.5)
            circlePatch_zoom = plt.Circle((0, 0), radius=circle_rad, fc='y',alpha=0.5)
            self.circle[name] = circlePatch
            self.circle_zoom[name] = circlePatch_zoom
            self.ax.add_patch(circlePatch)
            self.axins.add_patch(circlePatch_zoom)
            # Draw bands
            sectors = []
            sectors_zoom = []
            for i in range(10):
                ep = patches.Wedge((0,0),circle_rad,theta1=0,theta2=0,fill=True,alpha=0.6)
                ep_zoom = patches.Wedge((0,0),circle_rad,theta1=0,theta2=0,fill=True,alpha=0.6)
                sectors.append(ep)
                sectors_zoom.append(ep_zoom)
                self.ax.add_patch(ep)
                self.axins.add_patch(ep_zoom)
            
            self.bands_zoom[name] = sectors_zoom
            self.bands[name] = sectors

    def AddZone(self,xy,radius,color):
        circlePatch = patches.Arc((xy[0], xy[1]), width=2*radius,height =2*radius, fill =False, color=color)
        circlePatch_zoom = patches.Arc((xy[0], xy[1]), width=2*radius,height =2*radius, fill =False, color=color)
        self.ax.add_patch(circlePatch)
        self.axins.add_patch(circlePatch_zoom)

    def GetTriangle(self, tfsize, pos, vel, col):
        x = pos[0]
        y = pos[1]

        t = atan2(vel[1],vel[0])

        x1 = x + 2*tfsize * cos(t)
        y1 = y + 2*tfsize * sin(t)

        tempX = x - tfsize * cos(t)
        tempY = y - tfsize * sin(t)

        x2 = tempX + tfsize * cos((t + pi/2))
        y2 = tempY + tfsize * sin((t + pi/2))

        x3 = tempX - tfsize * cos((t + pi/2))
        y3 = tempY - tfsize * sin((t + pi/2))


        triangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], color=col, fill=True)
        #triangle.labelText = plt.text(x+2*tfsize, y+2*tfsize, "", fontsize=8)

        return triangle

    def UpdateTriangle(self,tfsize, pos, vel, poly):
        x = pos[0]
        y = pos[1]
        z = pos[2]

        t = atan2(vel[1], vel[0])

        x1 = x + 2*tfsize * cos(t)
        y1 = y + 2*tfsize * sin(t)

        tempX = x - 1*tfsize * cos(t)
        tempY = y - 1*tfsize * sin(t)

        x2 = tempX + 1*tfsize * cos((t + pi/2))
        y2 = tempY + 1*tfsize * sin((t + pi/2))

        x3 = tempX - 1*tfsize * cos((t + pi/2))
        y3 = tempY - 1*tfsize * sin((t + pi/2))

        poly.set_xy([[x1, y1], [x2, y2], [x3, y3]])
        #poly.labelText.set_position((x + 2*tfsize,y+2*tfsize))
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        #poly.labelText.set_text('Z:%.2f[m]\nS:%.2f[mps]' % (z,speed))

    def AddPath(self,path,color):
        if (path.shape[0] < 2):
            return
        if path.shape[1] == 10:
            tcps = path[:,4:7]
            tcpValues = path[:,7:10]
            if np.sum(tcps) > 0:
                from AccordUtil import plotTcpPlan
                n,e,d,ptn,pte,ptd = plotTcpPlan(path,tcps,tcpValues)
                self.wp.set_data(e,n,color)
                self.wp_zoom.set_data(e,n,color)
            else:
                self.wp.set_data(path[:,2],path[:,1])
                self.wp_zoom.set_data(path[:,2],path[:,1])
        else:
            self.wp.set_data(path[:,2],path[:,1])
            self.wp_zoom.set_data(path[:,2],path[:,1])
        #self.wp_scatter.set_array(np.array([path[:,2],path[:,1]]))
        #self.wp_scatter_zoom.set_array(np.array([path[:,2],path[:,1]]))

    def AddFence(self,fence,color):
        plt.plot(fence[:,1],fence[:,0],color)
        plt.scatter(fence[:,1],fence[:,0])
    
    def AddStartGoal(self):
        start_goal = [self.scenario.start_real[0:2],self.scenario.goal_real[0:2]]
        y,x = zip(*start_goal)
        self.sg, = self.ax.plot(x,y,color='yellow', marker='o',markersize=6,linestyle='')
        self.annot_start = self.ax.annotate('Start',
            xy=(x[0],y[0]), xycoords='data',
            xytext=(-70, -80), textcoords='offset points',
            size=8,
            bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=-90,angleB=0,rad=50"))
        self.annot_goal = self.ax.annotate('Goal',
            xy=(x[1],y[1]), xycoords='data',
            xytext=(-70, -80), textcoords='offset points',
            size=8,
            bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=-90,rad=50"))
        

    def UpdateStartGoal(self):
        start_goal = [self.scenario.start_real[0:2],self.scenario.goal_real[0:2]]
        y,x = zip(*start_goal)
        self.sg.set_data(x,y)
        self.annot_start.set_text('Start')
        self.annot_goal.set_text('Goal')

    def AddWaypoint(self):
        self.wp, = self.ax.plot(0,0,'k.-')
        self.wp_zoom, = self.axins.plot(0,0,'k.-')
        self.wp_scatter = self.ax.scatter(0,0)
        self.wp_scatter_zoom = self.axins.scatter(0,0)
    
    def UpdateWaypoint(self,i):
        #Global path 
        pln_aux = []
        cont = 0
        
        #for k, vec in enumerate(self.data['ownship0']['localPlans'][0]):
        #    if self.current_time >= vec[0] or vec[0] == 0.0:
        #        cont = cont + 1
        #    else:
        #        break
        
        #for k, vec in enumerate(self.data['ownship0']['localPlans'][0][0:cont+1]):
        #    vec[1] = self.data['ownship0']['localCoords'][k][0]
        #    vec[2] = self.data['ownship0']['localCoords'][k][1]
        #    pln_aux.append(vec)
        #self.AddPath(np.array(pln_aux),'k.-')


        idx = self.pathtime[i]+2
        if idx >= max(self.pathtime):
            idx = max(self.pathtime)-1

        for k, vec in enumerate(self.data['ownship0']['localPlans'][0][0:idx]):
            vec[1] = self.data['ownship0']['localCoords'][k][0]
            vec[2] = self.data['ownship0']['localCoords'][k][1]
            pln_aux.append(vec)
        self.AddPath(np.array(pln_aux),'k.-')
    
    def AddCostMap(self,costmap,xmin,xmax,ymin,ymax):
        #plt.gca().invert_yaxis()
        #plt.gca().invert_xaxis()
        self.costmap_y = np.linspace(ymin, ymax, len(costmap.y))
        self.costmap_x = np.linspace(xmin, xmax, len(costmap.x))
        self.costmap = costmap
        self.pcolormesh = self.ax.pcolormesh(self.costmap_x, self.costmap_y, self.costmap.z_time[self.costtime[0]]*0.01, cmap='rainbow', shading='nearest')
        #self.ax.add_patch(self.pcolormesh)

    def UpdateCostMap(self,i):
        z = self.costmap.z_time[self.costtime[i]]*0.01
        self.pcolormesh.set_array(z.ravel())
        

    def AddZoom(self):
        position = self.last_position
        self.pcolorzoom = self.axins.pcolormesh(self.costmap_x, self.costmap_y, self.costmap.z_time[self.costtime[0]]*0.01, cmap='rainbow', shading='nearest')
        x1, x2, y1, y2 = position[0]*1.000015, position[0]*0.999986,  position[1]*1.000024, position[1]*0.999975
        self.axins.set_xlim(x1, x2)
        self.axins.set_ylim(y1, y2)
        self.axins.set_xticklabels('')
        self.axins.set_yticklabels('')
        self.rectzoom = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',fill=False)
        self.ax.add_patch(self.rectzoom)
        #self.first_line, = self.ax.plot([x2,self.axins_pos[0][0]],[y2,self.axins_pos[0][1]],'k',linewidth=0.5)
        #self.second_line, = self.ax.plot([x2,self.axins_pos[1][0]],[y2,self.axins_pos[1][1]],'k',linewidth=0.5)
        #self.rectpatch, self.connects = self.ax.indicate_inset_zoom(self.axins)
        #self.ax.add_patch(self.rectpatch)

    def UpdateZoom(self,i):
        position = self.last_position
        x1, x2, y1, y2 = position[0]*1.000015, position[0]*0.999986,  position[1]*1.000024, position[1]*0.999975
        #self.first_line.set_data([x2,self.axins_pos[0][0]],[y2,self.axins_pos[0][1]])
        #self.second_line.set_data([x2,self.axins_pos[1][0]],[y2,self.axins_pos[1][1]])
        z = self.costmap.z_time[self.costtime[i]]*0.01
        self.pcolorzoom.set_array(z.ravel())
        self.axins.set_xlim(x1, x2)
        self.axins.set_ylim(y1, y2)
        self.axins.set_xticklabels('')
        self.axins.set_yticklabels('')
        self.rectzoom.set_xy((x1,y1))
        self.rectzoom.set_height(y2-y1)
        self.rectzoom.set_width(x2-x1)
        for j, vehicle in enumerate(self.agents_zoom):
                
            id = self.agentNames[j]
            #vehicle.center = (self.data[id][i][0], self.data[id][i][1])
            position = (self.data[id]["position"][i][1], self.data[id]["position"][i][0],self.data[id]["position"][i][2])
            velocity = (self.data[id]["velocityNED"][i][1], self.data[id]["velocityNED"][i][0])
            radius = self.agentsRadius[id]
            self.UpdateTriangle(radius,position,velocity,vehicle)
            self.paths_zoom[id].set_xdata(np.array(self.data[id]["position"])[:i,1])
            self.paths_zoom[id].set_ydata(np.array(self.data[id]["position"])[:i,0])

            
            if "trkbands" in self.data[id].keys():
                self.last_position = position
                if i < len(self.data[id]["trkbands"]):
                    if self.data[id]["trkbands"][i] != {}:
                        self.UpdateBands(position,self.data[id]["trkbands"][i],self.bands_zoom[id])
            if id in self.circle_zoom.keys():
                if self.circle_zoom[id] is not None:
                    self.circle_zoom[id].center = position
        
        #self.rectpatch, self.connects = self.ax.indicate_inset_zoom(self.axins)
    
    def AddObstacles(self,scenario):
        #Obstacle
        for polygon in scenario.obstacle_real:
            lat,lon = zip(*polygon[0])
            lat = list(lat)
            lon = list(lon)
            lat.append(polygon[0][0][0])
            lon.append(polygon[0][0][1])
        #    ax.plot(lon, lat, linestyle='-', color='red')
            self.obs, = self.ax.fill(lat, lon, facecolor='gray', edgecolor='black')
            self.obsaxins, = self.axins.fill(lat, lon, facecolor='gray', edgecolor='black')
    
    def UpdateObstacles(self):
        #Obstacle
        for polygon in self.scenario.obstacle_real:
            self.obs.set_xy(polygon[0])
            self.obs.set_fill(True)
            self.obsaxins.set_xy(polygon[0])
            self.obsaxins.set_fill(True)

    def UpdateBands(self,position,bands,sectors):
        numBands = bands["numBands"]
        low      = bands["low"]
        high     = bands["high"]
        btype    = bands["bandTypes"]
        h2c = lambda x:np.mod((360 -  (x - 90)),360)
        for sector in sectors:
            sector.set_theta1(0)
            sector.set_theta2(0)
            sector.set_center((0,0))
            sector.set_alpha(0)
        for i in range(numBands):
            max = h2c(low[i])
            min = h2c(high[i])
            if btype[i] != 1:
                if btype[i] == 4:
                    color = 'r'
                elif btype[i] == 5:
                    color = 'g'
                else:
                    color = 'b'
                sectors[i].set_center(position[:2])
                sectors[i].set_theta1(min)
                sectors[i].set_theta2(max)
                sectors[i].set_color(color)
                sectors[i].set_alpha(0.6)


    def init(self):
        return self.agents,self.paths,self.circle

    def animate(self,i):
        i = int(i*self.speed)
        self.current_time = self.current_time + 0.05
        if i < self.minlen-1:
            self.UpdateCostMap(i)
            self.UpdateObstacles()
            self.UpdateZoom(i)
            self.UpdateWaypoint(i)
            self.UpdateStartGoal()
            

            for j, vehicle in enumerate(self.agents):
                
                id = self.agentNames[j]
                #vehicle.center = (self.data[id][i][0], self.data[id][i][1])
                position = (self.data[id]["position"][i][1], self.data[id]["position"][i][0],self.data[id]["position"][i][2])
                velocity = (self.data[id]["velocityNED"][i][1], self.data[id]["velocityNED"][i][0])
                radius = self.agentsRadius[id]
                self.UpdateTriangle(radius,position,velocity,vehicle)
                self.paths[id].set_xdata(np.array(self.data[id]["position"])[:i,1])
                self.paths[id].set_ydata(np.array(self.data[id]["position"])[:i,0])

                if "trkbands" in self.data[id].keys():
                    self.last_position = position
                    if i < len(self.data[id]["trkbands"]):
                        if self.data[id]["trkbands"][i] != {}:
                            self.UpdateBands(position,self.data[id]["trkbands"][i],self.bands[id])
                if id in self.circle.keys():
                    if self.circle[id] is not None:
                        self.circle[id].center = position

        else:
            plt.close(self.fig)
        return self.agents,self.paths,self.circle

    def run(self):
        animate = lambda x: self.animate(x)
        init = lambda:self.init()
        self.anim = animation.FuncAnimation(self.fig, animate,
                                       init_func=init,
                                       frames=int(self.minlen/self.speed),
                                       interval=self.interval,
                                       repeat = False,
                                       blit=False)
        
        # Save animation as a movie
        if self.record:
            self.anim.save(self.filename, writer= "ffmpeg", fps=60)
        else:
            #plt.axis('off')
            plt.show()
