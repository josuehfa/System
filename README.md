
# Introduction
In recent years, the intense growth in the use of unmanned aircraft systems (UAS), which consists of one or more unmanned aircraft and the necessary elements for their operations, has caused changes in the relations that the society, the government and the market have with the aerospace industry. According to forecasts from the Aerospace Industries Association (AIA) and Avascent, unmanned aircraft will generate US$ 150 billion, in total expenditures, by 2036 and annually hold up to 60,000 research and development (R&D), manufacturing and services jobs. Due to the need to integrate these systems into the airspace in a safe manner, sharing this and producing low risks for third parties on the ground and cooperative and non-cooperative aircraft in operation. The UAS must comply with a series of requirements that aim to mitigate risks through the use of new computational systems. Among them, there is the use of Detect and Avoid System (DAA), which aims to see, feel or detect conflicting traffic and/or other dangerous conditions and take appropriate protective measures.

This work presents the development of a system for UAS trajectory planning. This system will optimize the trajectories generated and ensure safe operation with other aircraft on-air and third parties on the ground. In the developed system, the global planner is responsible for defining a flight plan, between two points on the airspace, for the UAS. This flight plan will optimize the trajectory costs concerning risk factors, such as population density, weather conditions and operation under restricted and/or dangerous zones. Moreover, a local planner will have the function of avoiding the risks related to conflicting air traffic during the operation of the UAS. It will use a DAA system in order to detect and avoid intruder aircraft risk of collision by performing preventive maneuvers calculated in the system after the violation of previously defined protection volumes.


## Simulation Results
The results of this project are shown in the next subtopics: 

### Local Path Planning using DAIDALUS
- Running DAIDALUS for a frontal collision between the ownership aircraft and an intruder aircraft. 

![Demo File](https://github.com/josuehfa/System/blob/master/simulation/frontalcolision.gif)

### Global Path Planning using OMPL

- Running the system considering costs of meteorological conditions, populational density and no-fly zones. 
  
![Demo File](https://github.com/josuehfa/System/blob/master/simulation/pathplanningresult.gif)


### Integrated scenario with Local and Global Path Planning

- Running the system considering the costs and risks related to collision with other aircraft and access to dangerous areas. 
  - Legend:
    - ![#000](https://via.placeholder.com/15/000/000000?text=+)  : Path planned
    - ![#ffff](https://via.placeholder.com/15/fff/000000?text=+) : Path executed by the UAS
    - ![#FF5733](https://via.placeholder.com/15/FF5733/000000?text=+) : Ownership Aircraft
    - ![#001ABA](https://via.placeholder.com/15/001ABA/000000?text=+) : Intruder Aircraft
    - ![#FFFC27](https://via.placeholder.com/15/FFFC27/000000?text=+) - ![#ED251B](https://via.placeholder.com/15/ED251B/000000?text=+) : DAA bands
    - ![#6C6C6C](https://via.placeholder.com/15/6C6C6C/000000?text=+) : No-fly zones
    - ![#E4EFF9](https://via.placeholder.com/15/E4EFF9/000000?text=+) - ![#330083](https://via.placeholder.com/15/330083/000000?text=+) : Costs
  

![Detect and Avoid Result](https://github.com/josuehfa/System/blob/master/simulation/finalresult_avoid.gif)


![Path Planning Result](https://github.com/josuehfa/System/blob/master/simulation/finalresult_path.gif)



# System Setup 
A system for path planning and navigation of UAS using ICAROUS, Ardupilot, RedeMet and others under MAVLink Protocol. 

### Install dependencies

`pip install -r requeriments.txt`
  
`sudo apt-get install gcc python3-dev libxml2-dev libxslt-dev`

`sudo apt-get install libwebkitgtk-dev libjpeg-dev libtiff-dev libgtk2.0-dev libsdl1.2-dev freeglut3 freeglut3-dev libnotify-dev libgstreamerd-3-dev`

`sudo apt-get install python-wxgtk3.0`

`sudo apt install python-opencv`

`sudo apt install python3-opencv`

`pip3 install pymavlink`


### Checkout ICAROUS
`git clone --recursive https://github.com/nasa/icarous.git`

#### Installing and Compilation
`cd icarous`
`bash UpdateModules.sh`
`make`
`make install`
  
#### StartUp:
`cd System/icarous/exe/cpu1`
`./core-cpu1 -C 1 -I 0`

### Checkout WebGS
`git clone --recursive https://github.com/nasa/webgs.git`
#### Installing dependencies:

 1. NodeJS and NPM 
 2. 
`curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -`
`sudo apt-get install -y nodejs`

#### Installing WebGS:
`cd webgs`
`./install.sh`

#### StartUp:

The simplest way:

`python3 start_webgs.py -DEV True`
 
See more options inside WebGS Repository

### Checkout ArduPilot
`git clone --recursive https://github.com/ArduPilot/ardupilot.git`

#### StartUp:
`cd System/icarous/Scripts`
`./runSITL.sh`


### Checkout PolyCARP
`git clone --recursive https://github.com/nasa/PolyCARP.git`

#### Installing
You need to add the PolyCarp python folder in PYTHONPATH to use it inside ICAROUS and MAVProxy.
`export PYTHONPATH="System/PolyCARP/Python"`


### Checkout MAVProxy
Following some tips about ICAROUS communication we need to use the release  1.8.20 
`git clone --recursive https://github.com/ArduPilot/MAVProxy.git`
`git checkout 6dd4a04`

#### Installing
Use the script inside ICAROUS to install MAVProxy
`cd System/icarous/Python/CustomModules`
`bash SetupMavProxy.sh System`
#### StartUp:
`cd System/icarous/Scripts`
`./runGS.sh`

### Checkout PyRedeMet

`git clone --recursive  https://github.com/josuehfa/pyredemet.git`

#### StartUp:
You need to create an account inside RedeMet and get an API there.

### OMPL
#### Download
OMPL script link here:
`https://ompl.kavrakilab.org/install-ompl-ubuntu.sh`
#### Installing
Make the script executable:
`chmod u+x install-ompl-ubuntu.sh`
  
Run the script to install OMPL with Python bindings.
`./install-ompl-ubuntu.sh --python`





## RUN EXAMPLE


#### load mission flight plan
wp load /home/josuehfa/System/icarous/Scripts/flightplan4.txt
#### load geofence
geofence load /home/josuehfa/System/icarous/Scripts/geofence2.xml
#### load traffic
traffic load 46 107 5 0 100 0

#### start mission from the home position
long MISSION_START






