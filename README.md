# System Setup 
A system for path planning and navigation of UAS using ICAROUS, Ardupilot, RedeMet and others under MAVLink Protocol. 

### Install dependencies

- `pip install -r requeriments.txt`
  
- `sudo apt-get install gcc python3-dev libxml2-dev libxslt-dev`

- `sudo apt-get install libwebkitgtk-dev libjpeg-dev libtiff-dev libgtk2.0-dev libsdl1.2-dev freeglut3 freeglut3-dev libnotify-dev libgstreamerd-3-dev`

- `sudo apt-get install python-wxgtk3.0`

- `sudo apt install python-opencv`

- `sudo apt install python3-opencv`

- `pip3 install pymavlink`


### Checkout ICAROUS
- `git clone --recursive https://github.com/nasa/icarous.git`

#### Installing and Compilation
- `cd icarous`
- `bash UpdateModules.sh`
- `make`
- `make install`
  
#### StartUp:
- `cd System/icarous/exe/cpu1`
- `./core-cpu1 -C 1 -I 0`

### Checkout WebGS
- `git clone --recursive https://github.com/nasa/webgs.git`
#### Installing dependencies:

 1. NodeJS and NPM 
- `curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -`
- `sudo apt-get install -y nodejs`
#### Installing WebGS:
- `cd webgs`
- `./install.sh`
#### StartUp:

The simplest way:

- `python3 start_webgs.py -DEV True`
 
See more options inside WebGS Repository

### Checkout ArduPilot
- `git clone --recursive https://github.com/ArduPilot/ardupilot.git`

#### StartUp:
- `cd System/icarous/Scripts`
- `./runSITL.sh`


### Checkout PolyCARP
- `git clone --recursive https://github.com/nasa/PolyCARP.git`

#### Installing
You need to add the PolyCarp python folder in PYTHONPATH to use it inside ICAROUS and MAVProxy.
- `export PYTHONPATH="System/PolyCARP/Python"`


### Checkout MAVProxy
Following some tips about ICAROUS communication we need to use the release  1.8.20 
- `git clone --recursive https://github.com/ArduPilot/MAVProxy.git`
- `git checkout 6dd4a04`

#### Installing
Use the script inside ICAROUS to install MAVProxy
- `cd System/icarous/Python/CustomModules`
- `bash SetupMavProxy.sh System`
#### StartUp:
- `cd System/icarous/Scripts`
- `./runGS.sh`

### Checkout PyRedeMet

- `git clone --recursive  https://github.com/josuehfa/pyredemet.git`

#### StartUp:
You need to create an account inside RedeMet and get an API there.

### OMPL
#### Download
OMPL script link here:
- `https://ompl.kavrakilab.org/install-ompl-ubuntu.sh`
#### Running
Make the script executable:
- `chmod u+x install-ompl-ubuntu.sh`
  
Run the script to install OMPL with Python bindings.
- `./install-ompl-ubuntu.sh --python`