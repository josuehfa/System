#!/bin/bash

MODE=active
SITL_HOST=127.0.0.1
SITL_INPUT_PORT=14551
COM_HOST=127.0.0.1
COM_INPUT_PORT=14552
COM_OUTPUT_PORT=14553
PX4_PORT=/dev/ttyACM0
PX4_BAUD=57600
#GS_MASTER=192.42.142.110:$COM_OUTPUT_PORT
GS_MASTER=127.0.0.1:$COM_OUTPUT_PORT
#GS_MASTER=/dev/ttyUSB0
GPIO_PORT=23
RADIO_SERIAL_PORT=/dev/ttyUSB0
RADIO_SOCKET_IN=$COM_OUTPUT_PORT
RADIO_SOCKET_OUT=$COM_INPUT_PORT
RADIO_BAUD=57600

echo "Launching Ground station test"
export PYTHONPATH="/home/josuehfa/System/PolyCARP/Python"
mavproxy.py --master=$GS_MASTER --map --console --load-module geofence,traffic --target-system=1 --target-component=5 --out=127.0.0.1:14550
