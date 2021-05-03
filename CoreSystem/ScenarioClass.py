import sys
import numpy as np
from MapGenClass import *
from skimage.draw import *

import math

radius_of_earth = 6378100.0

def distance(lat1, lon1, lat2, lon2):
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


class ScenarioClass():
    def __init__(self, scenario):
        '''
        Init Method:
            -  Define what scenario will be executed 
        '''
        if scenario == 'ONE':
            self.scenario = 'ONE'
            self.Scenario_One()
        elif scenario == 'TWO':
            self.scenario = 'TWO'
            self.Scenario_Two()
        elif scenario == 'THREE':
            self.scenario = 'THREE'
            self.Scenario_Three()
        elif scenario == 'FOUR':
            self.scenario = 'FOUR'
            self.Scenario_Four()
        elif scenario == 'FIVE':
            self.scenario = 'FIVE'
            self.Scenario_Five()
        
    def realToPorcent(self,point,region):
        '''
        Scale a point in a region to 0-1 range
        '''
        lat_region,lon_region = zip(*region)
        lat_region = list(lat_region)
        lon_region = list(lon_region)
        #Obter o intervalo de lat/lon da região
        lat_range = max(lat_region) -  min(lat_region)
        lon_range = max(lon_region) -  min(lon_region)
        #Lat/lon do pnt aleatoria menos a menor lat/lon
        lat_diff = point[0] - min(lat_region)
        lon_diff = point[1] - min(lon_region)
        #Porcentagem da diferença entre os range
        lat_cent = lat_diff/lat_range
        lon_cent = lon_diff/lon_range
        #Arrendondamento 
        point_new = (round(lon_cent,3),round(lat_cent,3))
        self.lat_region = lat_region
        self.lon_region = lon_region
        self.lat_range = lat_range
        self.lon_range = lon_range
        return point_new

    def generateObstacle(self):
        #Definição dos obstaculos
        base = 0
        topo = 0
        obstacle = []
        #Base Militar proxima a UFMG
        cpor = [(-19.870782, -43.959432),(-19.872528, -43.957651),
                (-19.877101, -43.963053),(-19.879381, -43.967742),
                (-19.877050, -43.970242)]
        #Centro de Desenvolvimento de Tecnologia Nuclear
        cdtn = [(-19.871252, -43.969874),(-19.872025, -43.969005),
                (-19.870834, -43.966205),(-19.871449, -43.965390),
                (-19.872377, -43.965454),(-19.874022, -43.967578),
                (-19.874012, -43.967868),(-19.874224, -43.967954),
                (-19.874476, -43.967785),(-19.875472, -43.969258),
                (-19.875772, -43.971406),(-19.872241, -43.971170)]
        #Estadio do Mineirão
        mineirao = [(-19.863243, -43.970448),(-19.863667, -43.971784),
                    (-19.865690, -43.972889),(-19.869151, -43.971661),
                    (-19.869080, -43.971318),(-19.866134, -43.969398)]
        #Arena do Independencia
        independencia = [(-19.909092, -43.918879),(-19.909385, -43.916926),
                         (-19.908094, -43.916733),(-19.907786, -43.918641)]
        #Aeroporto da Pampulha
        pampulha_airport = [(-19.844565, -43.965362),(-19.847330, -43.950663),
                            (-19.848571, -43.945384),(-19.853395, -43.936264),
                            (-19.857926, -43.936929),(-19.848129, -43.965994)]
        #Aeroporto Carlos Prates
        carlos_p_airport = [(-19.905402, -43.986880),(-19.910378, -43.993943),
                            (-19.911891, -43.993621),(-19.910698, -43.988835),
                            (-19.911283, -43.988631),(-19.912014, -43.985533),
                            (-19.913010, -43.984980),(-19.912120, -43.983269)]
        #Criando estrutura
        obstacle.append((cpor,base,topo,'CPOR/CMBH'))
        obstacle.append((cdtn,base,topo,'CDTN-UFMG'))
        obstacle.append((mineirao,base,topo,'Estadio Mineirão'))
        obstacle.append((independencia,base,topo,'Estadio Independencia'))
        obstacle.append((pampulha_airport,base,topo,'Aeroporto da Pampulha'))
        obstacle.append((carlos_p_airport,base,topo,'Aeroporto Carlos Prates'))
        return obstacle

    def Scenario_One(self):
        '''
        Primeiro cenario:
            Objetivo:
                -   Ir da EE-UFMG ate a Praça da Liberdade evitando as geofence out (safety)
            Otimização em relação a:
                -   Menor Caminho
                -   Geofence Out para os obstaculos definidos no inicio
        '''
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        #Praca da liberdade
        self.goal_real = (-19.931071, -43.937778,0)

        #test
        self.start_real = (-19.836548, -44.008020,0)
        self.goal_real = (-19.916969, -43.909915,0)

        #TESTFINAL
        self.start_real = (-19.854007, -43.984364)
        #self.goal_real = (-19.889572, -43.949868)
        self.goal_real = (-19.916489, -43.933327)

        #Região de Voo
        #self.region_real = [(-19.829752116279057, -44.02262249999998),
        #                    (-19.829752116279057, -43.90054215000001),
        #                    (-19.943540209302327, -43.90054215000001),
        #                    (-19.943540209302327, -44.02262249999998)]
        self.region_real = [(-19.833299760869554, -44.021236149999986),
                            (-19.833299760869554, -43.901276100000004),
                            (-19.942577826086957, -43.901276100000004),
                            (-19.942577826086957, -44.021236149999986)]


        #Geração dos Obstaculos
        self.obstacle_real = self.generateObstacle()

        #Scale start and goal to 0-1 range
        self.start = self.realToPorcent(self.start_real,self.region_real)
        self.goal = self.realToPorcent(self.goal_real,self.region_real)
        self.region = [(0, 0),(1, 0),(1, 1),(0, 1)]
        #Scale obstacle to 0-1 range
        self.obstacle = []
        for obs in self.obstacle_real:
            aux = []
            for point in obs[0]:
                aux.append(self.realToPorcent(point,self.region_real))
            self.obstacle.append((aux,obs[1],obs[2],obs[3]))
        
        #Parametros para criação do mapa
        self.time = 1
        self.nrows = 250
        self.ncols = 250
        self.mapgen = MapGen(self.nrows, self.ncols,self.time)
        self.mapgen.createScenarioOne()

        
    def Scenario_Two(self):
        '''
        Segundo cenario:
            Objetivo:
                -   Ir da EE-UFMG ate a Praça da Liberdade evitando as geofence out e passando pelas regiões com 
                menor densidade populacional (safety)
            Otimização em relação a:
                -   Menor Caminho
                -   Geofence Out para os obstaculos definidos no inicio
                -   Densidade Populacional obtida na imagem do IBGE
        '''
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        #Praca da liberdade
        self.goal_real = (-19.931071, -43.937778,0)

        #test
        self.start_real = (-19.836548, -44.008020,0)
        self.goal_real = (-19.916969, -43.909915,0)

        #TESTFINAL
        self.start_real = (-19.854007, -43.984364)
        #self.goal_real = (-19.889572, -43.949868)
        #Escola de Eng
        #self.start_real = (-19.869245, -43.963622,0) 
        self.goal_real = (-19.916489, -43.933327)

        #Região de Voo
        #self.region_real = [(-19.849635, -44.014423),(-19.849635, -43.900210),
        #                    (-19.934877, -43.900210),(-19.934877, -44.014423)]
        #self.region_real = [(-19.829752116279057, -44.02262249999998),
        #                    (-19.829752116279057, -43.90054215000001),
        #                    (-19.943540209302327, -43.90054215000001),
        #                    (-19.943540209302327, -44.02262249999998)]
        self.region_real = [(-19.833299760869554, -44.021236149999986),
                            (-19.833299760869554, -43.901276100000004),
                            (-19.942577826086957, -43.901276100000004),
                            (-19.942577826086957, -44.021236149999986)]

        
        #Geração dos Obstaculos
        #self.obstacle_real = self.generateObstacle()
        self.obstacle_real = []
        
        #Scale start and goal to 0-1 range
        self.start = self.realToPorcent(self.start_real,self.region_real)
        self.goal = self.realToPorcent(self.goal_real,self.region_real)
        self.region = [(0, 0),(1, 0),(1, 1),(0, 1)]
        #Scale obstacle to 0-1 range
        self.obstacle = []
        for obs in self.obstacle_real:
            aux = []
            for point in obs[0]:
                aux.append(self.realToPorcent(point,self.region_real))
            self.obstacle.append((aux,obs[1],obs[2],obs[3]))
        

        #Parametros para criação do mapa
        self.time = 1
        self.nrows = 200
        self.ncols = 200

        distlat = distance(self.region_real[0][0],self.region_real[0][1],self.region_real[3][0],self.region_real[3][1])
        distlon = distance(self.region_real[1][0],self.region_real[1][1],self.region_real[2][0],self.region_real[2][1])
        areamin = (distlat*distlon)/(self.ncols*self.ncols)

        
        
        self.mapgen = MapGen(self.nrows, self.ncols,self.time)
        self.mapgen.createScenarioTwo(areamin)

        
        


    def Scenario_Three(self):
        '''
        Terceiro cenario:
            Objetivo:
                -   Ir da EE-UFMG ate a Praça da Liberdade evitando as geofence out, passando pelas regiões com 
                menor densidade populacional (safety) e dentro do raio operacional ("ETOPS")
            Otimização em relação a:
                -   Menor Caminho
                -   Geofence Out para os obstaculos definidos no inicio
                -   Densidade Populacional obtida na imagem do IBGE
        '''
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        #Praca da liberdade
        self.goal_real = (-19.931071, -43.937778,0)

        #test
        self.start_real = (-19.836548, -44.008020,0)
        self.goal_real = (-19.916969, -43.909915,0)



        #Região de Voo
        #self.region_real = [(-19.849635, -44.014423),(-19.849635, -43.900210),
        #                    (-19.934877, -43.900210),(-19.934877, -44.014423)]
        self.region_real = [(-19.829752116279057, -44.02262249999998),
                            (-19.829752116279057, -43.90054215000001),
                            (-19.943540209302327, -43.90054215000001),
                            (-19.943540209302327, -44.02262249999998)]
        
        #Geração dos Obstaculos
        self.obstacle_real = self.generateObstacle()

        #Geração dos vertiports
        self.radius = 0.24
        self.vertiports_real = [(-19.887738, -44.015104),
                               (-19.836548, -44.008020),
                               (-19.919213, -43.992603),
                               (-19.935340, -43.949120),
                               (-19.916969, -43.909915)]

        #Scale start and goal to 0-1 range
        self.start = self.realToPorcent(self.start_real,self.region_real)
        self.goal = self.realToPorcent(self.goal_real,self.region_real)
        self.region = [(0, 0),(1, 0),(1, 1),(0, 1)]

        #Scale obstacle to 0-1 range
        self.obstacle = []
        for obs in self.obstacle_real:
            aux = []
            for point in obs[0]:
                aux.append(self.realToPorcent(point,self.region_real))
            self.obstacle.append((aux,obs[1],obs[2],obs[3]))

        #scale vertiports location
        self.vertiports = []
        for vertiport in self.vertiports_real:
            self.vertiports.append(self.realToPorcent(vertiport,self.region_real))

        #Parametros para criação do mapa
        self.time = 1
        self.nrows = 500
        self.ncols = 500
        self.mapgen = MapGen(self.nrows, self.ncols,self.time)
        self.mapgen.createScenarioThree(self.vertiports,self.radius)
    
    def Scenario_Four(self):
        '''
        Quarto cenario:
            Objetivo:
                -   Ir da EE-UFMG ate a Praça da Liberdade evitando as geofence out e evitando regioes com condições meteorologicas severas
            Otimização em relação a:
                -   Menor Caminho
                -   Geofence Out para os obstaculos definidos no inicio
                -   Condições meteorologicas
        '''
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        #Praca da liberdade
        self.goal_real = (-19.931071, -43.937778,0)

        #test
        self.start_real = (-19.836548, -44.008020,0)
        #proximo ao sta tereza
        self.goal_real = (-19.916969, -43.909915,0)
        #proximo a ufmg
        self.goal_real = (-19.877282, -43.954133,0)


        #TESTFINAL
        #self.start_real = (-19.854007, -43.984364)
        ##self.goal_real = (-19.889572, -43.949868)
        #self.goal_real = (-19.916489, -43.933327)

        #test
        #self.start_real = (-19.836548, -44.008020,0)
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        self.goal_real = (-19.916969, -43.909915,0)

        #Região de Voo
        #self.region_real = [(-19.849635, -44.014423),(-19.849635, -43.900210),
        #                    (-19.934877, -43.900210),(-19.934877, -44.014423)]
        #self.region_real = [(-19.829752116279057, -44.02262249999998),
        #                    (-19.829752116279057, -43.90054215000001),
        #                    (-19.943540209302327, -43.90054215000001),
        #                    (-19.943540209302327, -44.02262249999998)]
        self.region_real = [(-19.833299760869554, -44.021236149999986),
                            (-19.833299760869554, -43.901276100000004),
                            (-19.942577826086957, -43.901276100000004),
                            (-19.942577826086957, -44.021236149999986)]



        
        #Geração dos Obstaculos
        #self.obstacle_real = self.generateObstacle()
        self.obstacle_real = []
        #Geração dos vertiports
        #self.radius = 0.23
        #self.vertiports_real = [(-19.887738, -44.015104),
        #                       (-19.836548, -44.008020),
        #                       (-19.919213, -43.992603),
        #                       (-19.935340, -43.949120),
        #                       (-19.916969, -43.909915)]

        #Scale start and goal to 0-1 range
        self.start = self.realToPorcent(self.start_real,self.region_real)
        self.goal = self.realToPorcent(self.goal_real,self.region_real)
        self.region = [(0, 0),(1, 0),(1, 1),(0, 1)]

        #Scale obstacle to 0-1 range
        self.obstacle = []
        for obs in self.obstacle_real:
            aux = []
            for point in obs[0]:
                aux.append(self.realToPorcent(point,self.region_real))
            self.obstacle.append((aux,obs[1],obs[2],obs[3]))

        #scale vertiports location
        #self.vertiports = []
        #for vertiport in self.vertiports_real:
        #    self.vertiports.append(self.realToPorcent(vertiport,self.region_real))

        #Parametros para criação do mapa
        self.time = 20
        self.nrows = 200
        self.ncols = 200
        self.mapgen = MapGen(self.nrows, self.ncols,self.time)
        self.mapgen.createScenarioFour()
        pass

    def Scenario_Five(self):
        '''
        Quarto cenario:
            Objetivo:
                -   Ir da EE-UFMG ate a Praça da Liberdade evitando as geofence out, passando pelas regiões com 
                menor densidade populacional (safety) e evitando regioes com condições meteorologicas severas
            Otimização em relação a:
                -   Menor Caminho
                -   Geofence Out para os obstaculos definidos no inicio
                -   Densidade Populacional obtida na imagem do IBGE
                -   Condições meteorologicas
        '''
        #Escola de Eng
        self.start_real = (-19.869245, -43.963622,0) 
        #Praca da liberdade
        self.goal_real = (-19.931071, -43.937778,0)

        #test
        #self.start_real = (-19.836548, -44.008020,0)
        #self.goal_real = (-19.916969, -43.909915,0)
        
        #REAL !!!
        self.start_real = (-19.869245, -43.963622,0) 
        self.goal_real = (-19.916969, -43.909915,0)
        #!!!!

        # Chegada rapida para o pycarous
        #self.goal_real =(-19.871096, -43.950294)

        #Região de Voo
        #self.region_real = [(-19.849635, -44.014423),(-19.849635, -43.900210),
        #                    (-19.934877, -43.900210),(-19.934877, -44.014423)]
        #self.region_real = [(-19.829752116279057, -44.02262249999998),
        #                    (-19.829752116279057, -43.90054215000001),
        #                    (-19.943540209302327, -43.90054215000001),
        #                    (-19.943540209302327, -44.02262249999998)]
        
        self.region_real = [(-19.833299760869554, -44.021236149999986),
                            (-19.833299760869554, -43.901276100000004),
                            (-19.942577826086957, -43.901276100000004),
                            (-19.942577826086957, -44.021236149999986)]


        #Geração dos Obstaculos
        self.obstacle_real = self.generateObstacle()

        #Geração dos vertiports
        #self.radius = 0.23
        #self.vertiports_real = [(-19.887738, -44.015104),
        #                       (-19.836548, -44.008020),
        #                       (-19.919213, -43.992603),
        #                       (-19.935340, -43.949120),
        #                       (-19.916969, -43.909915)]

        #Scale start and goal to 0-1 range
        self.start = self.realToPorcent(self.start_real,self.region_real)
        self.goal = self.realToPorcent(self.goal_real,self.region_real)
        self.region = [(0, 0),(1, 0),(1, 1),(0, 1)]

        #Scale obstacle to 0-1 range
        self.obstacle = []
        for obs in self.obstacle_real:
            aux = []
            for point in obs[0]:
                aux.append(self.realToPorcent(point,self.region_real))
            self.obstacle.append((aux,obs[1],obs[2],obs[3]))

        #scale vertiports location
        #self.vertiports = []
        #for vertiport in self.vertiports_real:
        #    self.vertiports.append(self.realToPorcent(vertiport,self.region_real))

        #Parametros para criação do mapa
        self.time = 24
        self.nrows = 200
        self.ncols = 200

        distlat = distance(self.region_real[0][0],self.region_real[0][1],self.region_real[3][0],self.region_real[3][1])
        distlon = distance(self.region_real[1][0],self.region_real[1][1],self.region_real[2][0],self.region_real[2][1])
        areamin = (distlat*distlon)/(self.ncols*self.ncols)


        self.mapgen = MapGen(self.nrows, self.ncols,self.time)
        self.mapgen.createScenarioFive(areamin)
        pass

    def pathCost(self,path_x,path_y,scenario_time, type_):
                
        cost = 0
        for idx in range(len(scenario_time)-1):
            if (self.scenario == 'FIVE'):
                if type_ == 'pop':
                    costmap = self.mapgen.z_pop_time[scenario_time[idx]]

                    x1 = round(path_x[idx]*(costmap.shape[0]-1))
                    y1 = round(path_y[idx]*(costmap.shape[1]-1))
                    x2 = round(path_x[idx+1]*(costmap.shape[0]-1))
                    y2 = round(path_y[idx+1]*(costmap.shape[1]-1))
                    xx, yy = line(x1, y1, x2, y2)

                    for idy in range(len(xx)-1):
                        cost = cost + costmap[yy[idy+1]][xx[idy+1]]
                elif type_ == 'met':
                    costmap = self.mapgen.z_met_time[scenario_time[idx]]

                    x1 = round(path_x[idx]*(costmap.shape[0]-1))
                    y1 = round(path_y[idx]*(costmap.shape[1]-1))
                    x2 = round(path_x[idx+1]*(costmap.shape[0]-1))
                    y2 = round(path_y[idx+1]*(costmap.shape[1]-1))
                    xx, yy= line(x1, y1, x2, y2)

                    for idy in range(len(xx)-1):
                        cost = cost + costmap[yy[idy+1]][xx[idy+1]]
                elif type_ == 'ponderado':
                    costmap = self.mapgen.z_time[scenario_time[idx]]

                    x1 = round(path_x[idx]*(costmap.shape[0]-1))
                    y1 = round(path_y[idx]*(costmap.shape[1]-1))
                    x2 = round(path_x[idx+1]*(costmap.shape[0]-1))
                    y2 = round(path_y[idx+1]*(costmap.shape[1]-1))
                    xx, yy = line(x1, y1, x2, y2)

                    for idy in range(len(xx)-1):
                        cost = cost + costmap[yy[idy+1]][xx[idy+1]]

            else:
                costmap = self.mapgen.z_time[scenario_time[idx]]

                x1 = round(path_x[idx]*(costmap.shape[0]-1))
                y1 = round(path_y[idx]*(costmap.shape[1]-1))
                x2 = round(path_x[idx+1]*(costmap.shape[0]-1))
                y2 = round(path_y[idx+1]*(costmap.shape[1]-1))
                xx, yy = line(x1, y1, x2, y2)

                for idy in range(len(xx)-1):
                    cost = cost + costmap[yy[idy+1]][xx[idy+1]]
        
        self.pathcost = cost
        return cost
    
    def pathDist(self,path_x,path_y):
        
        dist = 0
        for idx in range(len(path_x)-1):
            dist = dist + distance(path_y[idx],path_x[idx],path_y[idx+1],path_x[idx+1])
        
        self.distcost = dist
        return dist

        



if __name__ == "__main__":

    scenario = ScenarioClass('FIVE')
    pass