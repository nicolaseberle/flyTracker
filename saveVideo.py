import os
import subprocess
from time import sleep

counter = 0
fd_out = '/media/pi/ext_drive/Recordings/'

while True:
    fout = fd_out + 'fly_tracker_1280_1080_' + str(counter) +  '.h264'
    print('save: ' + fout)
    cmd = 'raspivid -t 259200000 -w 1280 -h 1080 -fps 30 -g 10 -rf gray -b 17000000 -p 200,200,800,600 -ex off -hf -vf -o ' + fout 
    response = os.system(cmd)
    if response !=0:
        print(response)
        exit(0)
    sleep(0.001)
    counter = counter + 1
