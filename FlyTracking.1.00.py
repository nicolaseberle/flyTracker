

import cv2
import numpy as np
from multipleObjectTracker_3 import MultipleObjectTracker
from filevideostream import FileVideoStream
from genMosaic import GenMosaic
import tracker_constant as const
import argparse
import time

from tkinter import filedialog
import pandas as pd

import logging
import sys

class Parameters(object):
    def __init__(self,args):
        self.flag_hide_tracks = False
        self.init_once = False
        self.flag_init_record = False
    
        self.alpha_1 = 0.5 # transparency of the mask to see the tracks
        self.alpha_2 = 0 # transparency of the mask to see the tracks
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.color = np.random.randint(0,255,(255,3))
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.magic = args['magic']
        self.videofilename = args['input_video']
        self.display = args['no_preview']
        self.name_foo = const.DIR_WORKSPACE +  'fly_res_out.csv"
        
    def set_init_once(self,state):
        self.init_once = state

class Measurement(object):
    def __init__(self):
        self.numTot = 42
        headerTrack = ['Track_' + str(num) for num in range(self.numTot) ]
        self.header = pd.MultiIndex.from_product([headerTrack ,
                                     ['X','Y','VX','VY']],
                                    names=['track','pos'])
        self.pd_measurements = pd.DataFrame( columns = self.header ) 
        
    def saveMeasurement(self,tracker,numFrame,date):
        empty_array = np.empty((1,self.numTot*4,))
        empty_array [:] = np.nan
        tmp = pd.DataFrame(empty_array , index=[numFrame], columns = self.header )
        tmp.index.names = ['numFrame']
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                tmp.loc[numFrame,['Track_'+str(track.label)]] = [track.plot[0][0],track.plot[0][1],0,0]
        self.pd_measurements  = pd.concat([self.pd_measurements ,tmp])
        if numFrame%200 == 0:
            self.pd_measurements.to_csv(self.name_foo,mode='w',sep=':')
            print(self.pd_measurements.tail())
        
        
class Manager(object):
    def __init__(self,args):
        self.frame_full = None
        self.frame_date = None
        self.numFirstFrame = 30
        self.numCurrentFrame = 0
        self.measurement = Measurement()
        self.res = pd.DataFrame()
        self.parameters = Parameters(args)
        if self.parameters.magic == "1": 
            self.mosaic = GenMosaic()
        
    def process(self):
        self.calibration()
        self.initTracker()
        while(self.nextFrame()):
            self.extractionPlot()
            self.track()    
            self.save()
            self.display()
            
            if self.userControl():
               break
            
    def save(self):
        print("save Result")                
        self.measurement.saveMeasurement(self.tracker,self.numCurrentFrame,self.frame_date)
        
    def userControl(self):
        #keyboard control
        if self.parameters.display == 1:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                return 1
            #the key 'a' allows to fade out tracks 
            elif k==ord('a'):
                if self.parameters.flag_hide_tracks == False:
                    self.parameters.alpha_1 = 0.495
                    self.parameters.alpha_2 = -1
                    self.parameters.flag_hide_tracks = True
                    return 0
                else:
                    self.parameters.alpha_1 = 0.5
                    self.parameters.alpha_2 = 0
                    self.parameters.flag_hide_tracks = False
            elif k==-1:  # normally -1 returned,so don't print it
                return 0
            #the key "p" allows to have a break 
            elif k==ord('p'):
                while True:
                    c = cv2.waitKey(25)        
                    if c == ord('p'):
                        return 0
            elif k==ord('s'):
                if self.parameters.flag_init_record == False:
                    (h__, w__) = self.img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.out = cv2.VideoWriter("flyTracker_out.avi",fourcc , 30, (w__,h__))
                    self.parameters.flag_init_record = True
                else:
                    self.parameters.flag_init_record = False
                    self.out.release()
                    
            else:
                print(k) # else print its value
                
    def openVideo(self):
        print("openVideo")
        self.fvs = FileVideoStream(self.parameters.videofilename,1).start()
        time.sleep(1.0)
        if self.numFirstFrame>0:
            for counter in range(self.numFirstFrame):
                self.nextFrame()
                
        self.h,self.w = self.frame.shape[:2]
        
    def calibration(self):
        print("calibration")
        #Threshold calibration 
        self.findThresholdMin()
        
    def extractionPlot(self):
        print("extractionPlot")
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret,thresh2 = cv2.threshold(gray,self.minThreshold,255,cv2.THRESH_BINARY_INV)
        thresh2_dilate = cv2.dilate(thresh2,self.parameters.kernel,iterations = 1)
        thresh2_median  = cv2.medianBlur(thresh2_dilate,3)
    
        #if self.parameters.display == 1:
        #    cv2.imshow('maskMedian',thresh2_median)
            
        im2, contours, hierarchy = cv2.findContours(thresh2_median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        self.pos_t = np.empty((0,1,51), dtype='float32')
        
        #For each blob/contour 
        for cnt in contours:
            
            rect = cv2.minAreaRect(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            
            if perimeter > const.MAX_PERIMETER_BLOB:
                print("perimetre max atteint -> bordure génante")
                continue
            
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if self.parameters.magic == "1" :
                self.mosaic.addImage(self.frame[y:y+h, x:x+w])
            
            if len(cnt[:][:])>=5:
                ellipse = cv2.fitEllipse(cnt)
            else:
                ellipse = [(-1,-1),(-1,-1),-1]
            
            
            mask_2 = np.zeros_like(thresh2_median)
            cv2.fillPoly(mask_2, np.int_([cnt]), 1)
            
            #as we don't know if it's a cluster or not 
            #we compute different hypothesis 
            pixelpointsCV2 = cv2.findNonZero(mask_2[y:y+h, x:x+w])
            Z = np.float32(pixelpointsCV2 )
            number_points = len(Z)
            centroids = []
            for n in range(2,const.MAX_FLIES_BY_CLUSTER + 1):
                if number_points<=n:
                    cx = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4)
                    cy = ((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)
                    for num in range(n):
                        centroids.append(int(cx))
                        centroids.append(int(cy))
                        centroids.append(n)
                else:
                    ret,label,center = cv2.kmeans(Z,n,None,self.parameters.criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                    for num in range(n):
                        centroids.append(int(float(center[num][0]) + x))
                        centroids.append(int(float(center[num][1]) + y))
                        centroids.append(np.sum(label == num))
            
            
            self.frame  = cv2.drawContours(self.frame  ,[box],0,(0,255,0),1)

            cx = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4)
            cy = ((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)
            angle = np.arctan((box[0][1]-box[1][1])/(box[0][0]-box[1][0]))*180/np.pi
            
            
            centroids_array = np.asarray(centroids)
            centroids_array.astype(float)
            centroids_array = np.reshape(centroids_array,(1,1,-1))  
            
            current_plot = np.array([[[ cx,cy,angle*np.pi/180,area,ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2] ]]],dtype='float32')
            
            current_plot  = np.concatenate((current_plot , centroids_array), axis=2)
            
            #pos_t centroids of blob at time t
            self.pos_t = np.concatenate((self.pos_t,current_plot))
    
    
    def initTracker(self):
        print("initTracker")
        self.tracker = MultipleObjectTracker()
        
    def track(self):
        print("tracker")
        self.tracker.update_tracks(self.pos_t)
        
    def nextFrame(self):
        print("nextFrame : load next frame")
        err = self.fvs.more()
        if err !=1:
            return err    
        
        self.frame_date,self.frame = self.fvs.read()
        self.numCurrentFrame += 1
        return err
        
    def display(self):
        print("display")
        if not self.parameters.init_once:
            
            self.mask = np.zeros_like(self.frame)
            self.img = np.zeros_like(self.frame)
            
            self.parameters.set_init_once(True)
        
        print("time: ",str(self.frame_date), " num frame : " + str(self.numCurrentFrame))
        for i, track in enumerate(self.tracker.tracks):
            if track.has_match is True:
                print("PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))

                self.mask = cv2.line(self.mask, (int(track.old_plot[0][0]),int(track.old_plot[0][1])),(int(track.plot[0][0]),int(track.plot[0][1])), self.parameters.color[track.label].tolist(), 1)                
                cv2.putText(self.frame,str(track.label),(int(track.plot[0][0]),int(track.plot[0][1])), self.parameters.font, 0.4,(255,0,0),1,cv2.LINE_AA)
        cv2.putText(self.frame,'frame ' + str(self.numCurrentFrame),(10,20), self.parameters.font, 0.4,(255,0,0),1,cv2.LINE_AA)
        cv2.putText(self.frame,'nb tracks ' + str(len(self.tracker.tracks)),(10,40), self.parameters.font, 0.4,(255,0,0),1,cv2.LINE_AA)
        if self.parameters.flag_hide_tracks==True:
            cv2.putText(self.frame,"fade out activate",(10,60), self.parameters.font, 0.4,(255,0,0),1,cv2.LINE_AA)

        if self.parameters.flag_init_record == True:
            cv2.putText(self.frame,"save video",(10,80), self.parameters.font, 0.4,(255,0,0),1,cv2.LINE_AA)
            
        self.img = cv2.addWeighted(self.mask, 1, self.frame, 1, 0)
        cv2.addWeighted(self.mask, self.parameters.alpha_1, self.mask, 0.5, self.parameters.alpha_2 ,self.mask)
        
        if self.parameters.display == 1:
            cv2.imshow('res',self.img)
        
        if self.parameters.flag_init_record == True:
            self.out.write(self.img)

    def findThresholdMin(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 80;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 100
        params.minDistBetweenBlobs = 5
         
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1
         
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87
         
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01
        
        #use blob detector to establish the extraction threshold
        detector = cv2.SimpleBlobDetector_create(params)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Detect blobs.
        ex_mat = []
        keypoints = detector.detect(gray)
        for kpoint in keypoints:
            x = int(kpoint.pt[0])
            y = int(kpoint.pt[1])
            ex_mat.append(np.reshape(gray[y-10:y+10,x-10:x+10],(1,400)))
        
        print("blob statistics:",np.median(ex_mat),np.mean(ex_mat),np.percentile(ex_mat,20),np.std(ex_mat))
        self.minThreshold = np.median(ex_mat)-1.5*np.std(ex_mat)
        print("minThreshold : ",self.minThreshold)
        
    def stop(self):
        self.fvs.stop()

root = logging.getLogger()
root.setLevel(logging.DEBUG)
 
ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
#root.addHandler(ch)
out = None



def main(args):
    
    
    app = Manager(args)
    app.openVideo()
    app.process()    
    #init du tracker
    

    
    app.stop()


if __name__ == '__main__':
    __version__ = 1.0
    print("FlyTracker version  :" + str(__version__))
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-video", type=str, default=-1,
                    help="# relative path of the input video to analyse")
    ap.add_argument("-n", "--no-preview", type=str, default=1,
                    help="# desactivate the preview of the results")
    #not used
    ap.add_argument("-o", "--output", type=str, default=const.DIR_WORKSPACE,
                    help="# output directory")
    ap.add_argument("-m", "--magic", type=str, default=0,
                    help="# magic option")
                    
    args = vars(ap.parse_args())
    
    if args["input_video"] == -1:
        args["input_video"] = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("vid","*.h264"),("all files","*.*")))
        if len(args["input_video"])==0:
            print("no input file -> no analysis")
            exit(0)
    main(args)
