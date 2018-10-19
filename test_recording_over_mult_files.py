import picamera

fd_out = '/media/pi/ext_drive1/Recordings/'
camera = picamera.PiCamera(resolution=(1280, 1080))
camera.framerate = 30
camera.hflip=True
camera.vflip=True
camera.exposure_mode='off'
camera.start_recording(fd_out + 'seq_98.h264', format='h264', resize=None, splitter_port=1, bitrate=17000000,quality=20)
camera.wait_recording(3600)
for i in range(99, 220):
    #split_recording starts a new seq with a keyframe
    camera.split_recording(fd_out + 'seq_%d.h264' % i, format='h264', resize=None, splitter_port=1, bitrate=17000000,quality=20)
    camera.wait_recording(3600) 
camera.stop_recording()