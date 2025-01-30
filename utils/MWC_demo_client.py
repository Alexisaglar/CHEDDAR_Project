# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:12:41 2024

@author: Spencer
"""

import socket
import threading
import time
import struct
import json
from filelock import FileLock

# Connection data
own_wifi_addr = '192.168.1.16' # 16 Diana # 24 Spencer
server_wifi_addr = '192.168.1.19'

# wifi_addr = '127.0.0.1'

active = True
lock = FileLock("voltages_loads.json.lock")

rec_delay = 0.1

buffer_size = 1024

fmt = "!97f"

wifi_udp_port = 5555

def sock_init(): # Opening sockets
    udp_wifi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_wifi_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_wifi_sock.bind((own_wifi_addr, wifi_udp_port))
    
    print("UDP socket established!")
    
    return udp_wifi_sock

def udp_wifi_handle(sockname):
    while active:
        time.sleep(rec_delay)
        try:
            rec_data,_ = sockname.recvfrom(buffer_size)
            
            RTDS_data = list(struct.unpack(fmt, rec_data)) # 0-96 (1-97)
            
            RTDS_voltage_data = RTDS_data[0:33] # 0-32 (1-33)
            RTDS_active_load_data = [0] + RTDS_data[33:65] # 33-64 (34-65)
            RTDS_reactive_load_data = [0] + RTDS_data[65:97] # 65-96 (67-99)
            
        except:
            print("Error with receiving UDP data.")
            
        try:
            json_data = {}
            
            for i, voltage in enumerate(RTDS_voltage_data,start=1):
                json_data[i] = {
                    "voltage": voltage,
                    "active": RTDS_active_load_data[i-1],
                    "reactive": RTDS_reactive_load_data[i-1]
                }

            print(f"Thread {threading.current_thread().name} writing to json...")
            with lock:
                with open("voltages_loads.json", "w") as json_file:
                    json.dump(json_data, json_file, indent=4)
            print(f"Thread {threading.current_thread().name} finished writing.")
        except:
            print("Error with writing json file.")

udp_wifi_sock = sock_init()

udp_wifi = threading.Thread(target=udp_wifi_handle,
                           args=(udp_wifi_sock,),
                           daemon=True)
udp_wifi.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    active = False
    udp_wifi_sock.close()
    print("Stopping...")
