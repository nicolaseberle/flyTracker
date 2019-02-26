

import cv2
import numpy as np
from multipleObjectTracker_3 import MultipleObjectTracker
from filevideostream import FileVideoStream
from genMosaic import GenMosaic
import tracker_constant as const
import argparse
import time
import csv
from QRcodeDetection import QRCodeDetector,AreneDetector

from tkinter import filedialog
import pandas as pd

import logging
import sys

from joblib import Parallel, delayed
import multiprocessing

class Parameters(object):
    def __init__(self,args,num_arene):
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
        self.num_arene = num_arene
        self.name_foo = []
        self.ofile = []
        self.writer = []
        #for label in range (20):
        #    self.name_foo.append(const.DIR_WORKSPACE +  'fly_arene_' + str(num_arene) + '_num_' + str(label) + '.csv')
        #    self.createFoo(label)

    def set_init_once(self,state):
        self.init_once = state

    def createFoo(self,label):
        self.ofile.append(open(str(self.name_foo[label]),"w"))
        self.writer.append( csv.writer(self.ofile[label], delimiter=':'))
        self.writer[label].writerow(["Date(ms)","numFrame","X (in pixel)","Y (in pixel)","VX (in pixel/s)","VY (in pixel/s)",'Touch (1) / Alone (0)'])#Alone or Touch

class Measurement(object):
    def __init__(self,param):
        self.numTot = 40
        self.name_foo = param.name_foo
        headerTrack = ['Track_' + str(num) for num in range(self.numTot) ]
        self.header = pd.MultiIndex.from_product([headerTrack ,
                                     ['X','Y','VX','VY','Theta','T']],
                                    names=['track','pos'])
        self.pd_measurements = pd.DataFrame( columns = self.header )
        self.parameters = param

    def saveMeasurementDF(self,tracker,numFrame,date):
        empty_array = np.empty((1,self.numTot*6,))
        empty_array [:] = np.nan
        tmp = pd.DataFrame(empty_array , index=[numFrame], columns = self.header )
        tmp.index.names = ['numFrame']
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                tmp.loc[numFrame,['Track_'+str(track.label)]] = [track.plot[0][0],track.plot[0][1],track.speed[0][0],track.speed[0][1],track.plot[0][2],track.flag_touch]

        self.pd_measurements  = pd.concat([self.pd_measurements ,tmp])
        if numFrame%200 == 0:
            self.pd_measurements.to_csv(self.name_foo,mode='w',sep=':')

    def saveMeasurementCSV(self,tracker,numFrame,date):
        for i, track in enumerate(tracker.tracks):
            if len(self.parameters.writer)>track.label:
                self.parameters.writer[track.label].writerow([date,numFrame , track.plot[0][0],track.plot[0][1],track.speed[0][0], track.speed[0][1],track.flag_touch ] )
            else:
                self.parameters.name_foo.append(const.DIR_WORKSPACE +  'fly_arene_' + str(self.parameters.num_arene) + '_num_' + str(track.label) + '.csv')
                self.parameters.ofile.append(open(str(self.parameters.name_foo[track.label]),"w"))
                self.parameters.writer.append(csv.writer(self.parameters.ofile[track.label], delimiter=':'))
                self.parameters.writer[track.label].writerow(["Date(ms)","numFrame","X (in pixel)","Y (in pixel)","VX (in pixel/s)","VY (in pixel/s)",'Touch (1) / Alone (0)'])#Alone or Touch
                self.parameters.writer[track.label].writerow([date,numFrame , track.plot[0][0],track.plot[0][1],track.speed[0][0], track.speed[0][1],track.flag_touch ] )

class Manager(object):
    def __init__(self,args,roi):
        self.frame_full = None
        self.frame_date = None
        self.numFirstFrame = 30
        self.numCurrentFrame = 0
        self.res = pd.DataFrame()
        self.parameters = Parameters(args,roi[0])
        self.measurement = Measurement(self.parameters)
        self.roi = roi
        self.numArene = roi[0]
        self.initDisplay = 0
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
            #self.signalToUpdatePosArene()
            if self.userControl():
               break

    def signalToUpdatePosArene(self):
        if self.numCurrentFrame%100 == 0:
            self.fvs.updateRoi(self.roi)
            self.initPosArene()

    def save(self):
        #self.measurement.saveMeasurementDF(self.tracker,self.numCurrentFrame,self.frame_date)
        self.measurement.saveMeasurementCSV(self.tracker,self.numCurrentFrame,self.frame_date)

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
        self.fvs = FileVideoStream(self.parameters.videofilename,1,64,self.roi).start()
        time.sleep(1.0)
        if self.numFirstFrame>0:
            for counter in range(self.numFirstFrame):
                self.nextFrame()

        self.h,self.w = self.frame.shape[:2]


    def calibration(self):
        print("calibration")
        #Threshold calibration
        #self.findThresholdMin_2()
        self.findThresholdMin()

    def  deleteQRcode(self,thresh_frame):
        print("delete QR code")
        marge = 50
        h,w = thresh_frame.shape[:2]
        thresh_frame[0:marge,0:marge] = 0
        thresh_frame[h-marge:,w-marge:] = 0
        thresh_frame[h-marge:,0:marge] = 0
        thresh_frame[0:marge, w-marge:,] = 0
        return thresh_frame

    def extractionPlot(self):
        print("extractionPlot")
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret,thresh2 = cv2.threshold(gray,self.minThreshold,255,cv2.THRESH_BINARY_INV)
        thresh2 = self.deleteQRcode(thresh2)
        thresh2_dilate = cv2.dilate(thresh2,self.parameters.kernel,iterations = 1)
        thresh2_median  = cv2.medianBlur(thresh2_dilate,3)

        #if self.parameters.display == 1:
        #    cv2.imshow('maskMedian',thresh2_median)

        im2, contours, hierarchy = cv2.findContours(thresh2_median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        self.pos_t = np.empty((0,1,51), dtype='float32')
        self.init_brief()

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


        self.compute_brief_ORB(gray,self.pos_t)
    #def brief(self,image):
    #    self.hist = self.hog.compute(image,self.winStride,self.padding,self.locations)

    def compute_brief_ORB(self, img,list_pts):
        kpts = []
        kpts.extend([cv2.KeyPoint(int(pt[0][0]), int(pt[0][1]), 1) for pt in list_pts])
        self.kp, self.descr = self.orb.compute(img, kpts)
        #print(kpts,self.kp)
        #print(list_pts.shape,len(kpts),len(self.kp),self.descr.shape)
        #print(self.descr)

    def findMatches(self):
        self.matches = self.matcher.match(self.old_descr, self.descr)

    def init_brief(self):
        self.orb = cv2.ORB_create()
        self.init_matcher()

    def init_matcher(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def initTracker(self):
        print("initTracker")
        self.tracker = MultipleObjectTracker()

    def track(self):
        print("tracker")
        self.tracker.update_tracks(self.pos_t,self.frame_date)

    def nextFrame(self):
        print("nextFrame : load next frame")
        err = self.fvs.more()
        time.sleep(0.020)
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
                print("ARENE " + str(self.numArene) + " PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))
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

        if self.parameters.display == 1: #and  self.numCurrentFrame%2==0:
            figureName = 'res_' + str(self.numArene)
            cv2.imshow(figureName,self.img)
            cv2.waitKey(1)

        if self.parameters.flag_init_record == True:
            self.out.write(self.img)

    def findThresholdMin_2(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        surface = []
        for iter_thres in range(0,255):
            ret,mask_thresh = cv2.threshold(gray,iter_thres,255,cv2.THRESH_BINARY_INV)
            pixelpointsCV2 = cv2.findNonZero(mask_thresh)
            Z = np.float32(pixelpointsCV2 )

            if (np.isnan(Z)).any():
                surface.append(0)
            else:
                surface.append(len(Z))

        print(surface)


    def findThresholdMin(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        marge  = 5

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
        gray = self.deleteQRcode(gray)

        # Detect blobs.
        ex_mat = []
        keypoints = detector.detect(gray)
        #print(keypoints)
        height,width = gray.shape
        if len(keypoints) != 0:
            for kpoint in keypoints:
                x = int(kpoint.pt[0])
                y = int(kpoint.pt[1])
                if x > 2*marge and y > 2*marge and x < width - 2*marge and y < height - 2*marge:
                    ex_mat.append(np.reshape(gray[y-marge:y+marge,x-marge:x+marge],(1,100)))

            print("blob statistics:",np.median(ex_mat),np.mean(ex_mat),np.percentile(ex_mat,20),np.std(ex_mat))
            self.minThreshold = np.median(ex_mat)-np.std(ex_mat)
            print("minThreshold : ",self.minThreshold)
        else:
            print("no blob has been found")
            self.minThreshold = 40
            print("minThreshold : ",self.minThreshold)

    def stop(self):
        self.fvs.stop()

    def initPosArene(self):
        posArene = []
        height,width,channel = self.frame.shape
        scale = 2
        src = cv2.resize(self.frame,(int(width/scale), int(height/scale)), interpolation = cv2.INTER_LINEAR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=20, param2=50,
                               minRadius=int(80/scale), maxRadius=int(140/scale))
        index_roi = 1
        for i in circles[0, :]:
            x1 = (int(i[0]*scale-200), int(i[1]*scale-200))
            x2 = (int(i[0]*scale+200), int(i[1]*scale+200))
            str_roi = index_roi
            posArene.append([index_roi,x1,x2])
            logging.info('Arene %s x:%d y:%d r:%d',index_roi,x1[0],x1[1],200)
            index_roi = index_roi +1
            # draw the center of the circle
            cv2.circle(src,(i[0],i[1]),2,(0,255,0),3)
            cv2.imshow('detected circles',src)
            cv2.waitKey(0)

        return posArene

out = None

class MultiAppManager(object):
    def __init__(self,args,posArene):
        self.args = args
        self.nb_arene = len(posArene)
        self.posArene = posArene

    def run(self,roi):
        app = Manager(self.args,roi)
        app.openVideo()
        app.process()
        app.stop()

def initPosArene(args,flag):
# function to initialize the position of each arena. if no pattern, there are no initialization
# flag 0: 'no'
# flag 1: 'circle'
# flag 2: 'qrcode'
    posArene = []
    fvs = FileVideoStream(args['input_video'],1).start()
    time.sleep(1.0)
    err = fvs.more()
    date,frame = fvs.read()
    height,width,channel = frame.shape
    # automatic extraction of arean circles
    # TO DO - param1 param2 have to be defined automatically
    if flag == 1:
        scale = 2.5
        src = cv2.resize(frame,(int(width/scale), int(height/scale)), interpolation = cv2.INTER_LINEAR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=20, param2=50,
                               minRadius=int(80/scale), maxRadius=int(140/scale))
        if circles.shape[1] == 4:
            index_roi = 1
            for i in circles[0, :]:
                x1 = (int(i[0]*scale-200), int(i[1]*scale-200))
                x2 = (int(i[0]*scale+200), int(i[1]*scale+200))
                str_roi = index_roi
                posArene.append([index_roi,x1,x2])
                logging.info('Arene %s x:%d y:%d r:%d',index_roi,x1[0],x1[1],200)
                index_roi = index_roi +1
    #automatic extraction of the arena thanks to QRCode
    elif flag == 2:
        QRCode = QRCodeDetector(frame)
        err = QRCode.scan()
        print(QRCode.getPattern())
        QRCode.display()
        if err > 0:
            logging.warning('erreur detection arene')
            exit(0)

        Arene = AreneDetector(QRCode.getPattern(),frame)
        posArene = Arene.computeArenePos()
        Arene.display()

    return posArene

def main(args):
    posArene = []
    if args["detectionArene"] == 'circle':
        num_cores = 4#multiprocessing.cpu_count()
        posArene = initPosArene(args,1)
    elif args["detectionArene"] == 'no':
        num_cores = 1#multiprocessing.cpu_count()
        posArene.append([1,(int(1), int(1)),(int(1200), int(1000))])
        #open one frame -> select manually arena
        #
        #TO TO
        #
    elif args["detectionArene"] == 'qrcode':
        num_cores = 4#multiprocessing.cpu_count()
        posArene = initPosArene(args,2)
        print(posArene)
    else:
        num_cores = 1#multiprocessing.cpu_count()
        posArene.append([1,(int(1), int(1)),(int(1200), int(1000))])

    if posArene == []:
        num_cores = 1#multiprocessing.cpu_count()
        posArene.append([1,(int(1), int(1)),(int(1200), int(1000))])
    multiApp = MultiAppManager(args,posArene)
    results = Parallel(n_jobs=num_cores)(delayed(multiApp.run)(roi) for roi in posArene)

if __name__ == '__main__':

    __version__ = 2.0
    print("FlyTracker version  :" + str(__version__))
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-video", type=str, default=-1,
                    help="# relative path of the input video to analyse")
    ap.add_argument("-n", "--no-preview", type=str, default=1,
                    help="# desactivate the preview of the results")
    ap.add_argument("-o", "--output", type=str, default=const.DIR_WORKSPACE,
                    help="# output directory")
    ap.add_argument("-m", "--magic", type=str, default=0,
                    help="# magic option")
    ap.add_argument("-d", "--detectionArene", type=str, default='qrcode',
                help="# detection auto des arene no|auto|circle|qrcode")
    args = vars(ap.parse_args())

    logging.basicConfig(filename=args['output'] + '/info.log',format='%(asctime)s %(threadName)s %(levelname)-8s %(message)s',level=logging.DEBUG,filemode='w')
    logging.info('FlyTracker version: %s', str(__version__))
    logging.info('input video -i: %s', args['input_video'])
    logging.info('output directory: %s', args['output'])
    logging.info('Arena detection: %s', args['detectionArene'])

    if args["input_video"] == -1:
        args["input_video"] = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("vid","*.h264"),("all files","*.*")))
        logging.info('input video dial: %s', args['input_video'])
        if len(args["input_video"])==0:
            print("no input file -> no analysis")
            logging.warning("no input file -> no analysis")
            exit(0)

    main(args)
