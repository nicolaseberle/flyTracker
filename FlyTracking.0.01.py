

import cv2
import numpy as np
from multipleObjectTracker_3 import MultipleObjectTracker
from filevideostream import FileVideoStream
from genMosaic import GenMosaic
import tracker_constant as const
import argparse
import time

from tkinter import filedialog

import logging
import sys


def findThresholdMin(frame):
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect blobs.
    ex_mat = []
    keypoints = detector.detect(gray)
    for kpoint in keypoints:
        x = int(kpoint.pt[0])
        y = int(kpoint.pt[1])
        ex_mat.append(np.reshape(gray[y-10:y+10,x-10:x+10],(1,400)))
    
    print("blob statistics:",np.median(ex_mat),np.mean(ex_mat),np.percentile(ex_mat,20),np.std(ex_mat))
    new_threshold = np.median(ex_mat)-np.std(ex_mat)
    print("new_threshold : ",new_threshold)
    
    return new_threshold
    #gray[keypoints[0], x:x+w]
    
    # Draw detected blobs as red circles.
    #frame2= np.copy(frame)
    #cv2.drawKeypoints(gray,keypoints,frame2,color=(0,255,0), flags=0)
    

root = logging.getLogger()
root.setLevel(logging.DEBUG)
 
ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
#root.addHandler(ch)

def main(args):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    flag_hide_tracks = False
    init_once = False
    
    alpha_1 = 0.5 # transparency of the mask to see the tracks
    alpha_2 = 0 # transparency of the mask to see the tracks
    #X vers le bas, Y vers la droite
    
    #kernel = np.ones((3,3),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    color = np.random.randint(0,255,(255,3))
    
#    lk_params = dict( winSize  = (20,20),
#                       maxLevel = 1,
#                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    
    ############################################################################
    fvs = FileVideoStream(args['input_video'],1).start()
    time.sleep(1.0)
    seuil_bas=70
    ############################################################################    

    
        
    for counter in range(50):
        fvs.more()
        frame_full = fvs.read()
    numFrame = 50    
    h,w = frame_full.shape[:2]
    
    #Threshold calibration 
    seuil_bas = findThresholdMin(frame_full)
    
    #init du tracker
    tracker = MultipleObjectTracker()
    if args['magic'] == "1":
        mosaic = GenMosaic()
    
   
    while fvs.more():
        print('***************************************************************')
        # Take each frame
        frame = fvs.read()
        
        # Convert BGR to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             
        if not init_once:
            
            mask = np.zeros_like(frame)
            img = np.zeros_like(frame)
            
            init_once = True

        # find the keypoints with ORB
        #gray2 = cv2.resize(gray,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        #frame2 = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        #current_kp, current_desc = orb.detectAndCompute(gray2, None)
        # compute the descriptors with ORB
        
        #cv2.drawKeypoints(gray2,current_kp,frame2,color=(0,255,0), flags=0)
        #if args['no_preview'] == 1:
        #    cv2.imshow('frameRGB',frame2)
        
        ret,thresh2 = cv2.threshold(gray,seuil_bas,255,cv2.THRESH_BINARY_INV)
        thresh2_dilate = cv2.dilate(thresh2,kernel,iterations = 1)
        thresh2_median  = cv2.medianBlur(thresh2_dilate,3)
    
        if args['no_preview'] == 1:
            cv2.imshow('maskMedian',thresh2_median)
            #cv2.imshow('unknow',unknown)
            
        im2, contours, hierarchy = cv2.findContours(thresh2_median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        pos_t = np.empty((0,1,51), dtype='float32')
        
        #For each blob/contour 
        for cnt in contours:
            
            rect = cv2.minAreaRect(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            #if the perimeteter is too high, then this blob is ignored 
            if perimeter > const.MAX_PERIMETER_BLOB:
                print("perimetre max atteint -> bordure génante")
                continue
            #else this blob is a plot
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if args['magic'] == "1" :
                mosaic.addImage(frame[y:y+h, x:x+w])
            
            if len(cnt[:][:])>=5:
                # Bounding ellipse
                
                
                #easy function
                ellipse = cv2.fitEllipse(cnt)
                #print(ellipse)
                #add it
                #cv2.ellipse(img_2, ellipse, (0,255,0), 2,cv2.LINE_AA)
                #cv2.imshow("ellispe",img_2)
            else:
                ellipse = [(-1,-1),(-1,-1),-1]
            
            
            mask_2 = np.zeros_like(thresh2_median)
            cv2.fillPoly(mask_2, np.int_([cnt]), 1)
            
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
                    ret,label,center = cv2.kmeans(Z,n,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                    for num in range(n):
                        centroids.append(int(float(center[num][0]) + x))
                        centroids.append(int(float(center[num][1]) + y))
                        centroids.append(np.sum(label == num))
            
            
            frame = cv2.circle(frame,(int(centroids[0]),int(centroids[1])), 2, (255,255,255), -1)
            frame = cv2.circle(frame,(int(centroids[3]),int(centroids[4])), 2, (255,255,255), -1)
            
            #print(1,nb_element,area, area/nb_element)
            
            if(M['m00']!=0):
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                angle = 0
            else:
                cx = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4)
                cy = ((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)
                angle = 0
                
                
            
            centroids_array = np.asarray(centroids)
            centroids_array.astype(float)
            centroids_array = np.reshape(centroids_array,(1,1,-1))  
            
            current_plot = np.array([[[ cx,cy,angle*np.pi/180,area,ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2] ]]],dtype='float32')
            
            current_plot  = np.concatenate((current_plot , centroids_array), axis=2)
            
            #pos_t centroids of blob at time t
            pos_t = np.concatenate((pos_t,current_plot))
        #update of the tracker with new plots
        tracker.update_tracks(pos_t)
        
        print("num frame : " + str(numFrame))
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                print("PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))
                #mask = cv2.circle(mask,(track.plot[0][0],track.plot[0][1]), 2, color[track.label].tolist(), -1)
                mask = cv2.line(mask, (int(track.old_plot[0][0]),int(track.old_plot[0][1])),(int(track.plot[0][0]),int(track.plot[0][1])), color[track.label].tolist(), 1)
                #mask = cv2.line(mask, (int(track.plot[0][0]),int(track.plot[0][1])),(int(track.plot[0][0]+track.speed[0][0]*(1/30)),int(track.plot[0][1]+track.speed[0][1]*(1/30))), (255 ,0 ,0), 1)
                #mask = cv2.circle(mask,(int(track.old_plot[0][0]),int(track.old_plot[0][1])), 2, (255,0,255), -1)
                #mask = cv2.circle(mask,(int(track.plot[0][0]),int(track.plot[0][1])), 2, (0,255,0), -1)
                cv2.putText(frame,str(track.label),(int(track.plot[0][0]),int(track.plot[0][1])), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            #if track.has_match is False:
                #cv2.putText(frame,str(track.label),(track.plot[0][0],track.plot[0][1]), font, 0.4,(48, 214, 232),1,cv2.LINE_AA)
        cv2.putText(frame,'frame ' + str(numFrame),(10,20), font, 0.4,(255,0,0),1,cv2.LINE_AA)
        
        if flag_hide_tracks==True:
            cv2.putText(frame,"fade out activate",(10,40), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            
        img = cv2.addWeighted(mask, 1, frame, 1, 0)
        cv2.addWeighted(mask, alpha_1, mask, 0.5, alpha_2 ,mask)
        
        if args['no_preview'] == 1:
            cv2.imshow('res',img)
        
        numFrame = numFrame + 1
        

        #keyboard control
        if args['no_preview'] == 1:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
            #the key 'a' allows to fade out tracks 
            elif k==ord('a'):
                if flag_hide_tracks == False:
                    alpha_1 = 0.495
                    alpha_2 = -1
                    flag_hide_tracks = True
                    continue
                else:
                    alpha_1 = 0.5
                    alpha_2 = 0
                    flag_hide_tracks = False
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            #the key "p" allows to have a break 
            elif k==ord('p'):
                while True:
                    c = cv2.waitKey(25)        
                    if c == ord('p'):
                        break
            else:
                print(k) # else print its value
        
    fvs.stop()


if __name__ == '__main__':
    __version__ = 0.3
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
