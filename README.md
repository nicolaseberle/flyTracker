#######################################################################
Solution 1

Terminal 1
raspivid -t 0 -w 640 -h 480 -hf -ih -fps 20 -n -o - | nc  localhost 2222

Terminal 2
nc -l 2222 | mplayer -fps 200 -demuxer h264es -

#######################################################################
Solution 2
Avantage: on a un delay entre l'acquisition et le traitement mais on traite toutes les images.  

Terminal 1
raspivid -t 0 -w 640 -h 480 -hf -ih -fps 30 -n -o - | nc  localhost 5000

Terminal 2
nc -l -p 5000 -v > fifo

Terminal 3
python ~/workspace/hello-websocket-master/NetCatFifoViewer.py

#######################################################################
Solution 3
Avantage : Meilleur Live, en revanche on perd beaucoup d'image. 16 FPS ACQ. 12 FPS TRAITEMENT

cd ~/workspace/hello-websocket-master/

Terminal 1
make recorder

Termianl 2
make server

lancer une page html://localhost:9000
ça lancera l'acquisition des images

#######################################################################

raspivid -t 10000 -w 1280 -h 960 -fps 30 -ex auto -rf gray -b 1200000 -p 200,200,640,480 -o fly_tracker.h264 && MP4Box -add fly_tracker.h264 fly_tracker.mp4