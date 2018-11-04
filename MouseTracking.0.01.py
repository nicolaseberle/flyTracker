# coding=utf-8

import cv2
import numpy as np
from multipleObjectTracker_4 import MultipleObjectTracker
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
    params.maxThreshold = 140;
     
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
    new_threshold = np.median(ex_mat)-2*np.std(ex_mat)
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
out = None

def main(args):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    flag_hide_tracks = False
    init_once = False
    flag_init_record = False
    
    alpha_1 = 0.5 # transparency of the mask to see the tracks
    alpha_2 = 0 # transparency of the mask to see the tracks
    #X vers le bas, Y vers la droite
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    color = np.random.randint(0,255,(255,3))
        
    ############################################################################
    fvs = FileVideoStream(args['input_video'],1).start()
    time.sleep(1.0)
    ############################################################################    
    numFrame = 5
    for counter in range(numFrame):
        fvs.more()
        frame_pos,frame_full = fvs.read()
    
    h,w = frame_full.shape[:2]
    
    #Threshold calibration 
    minThreshold = 60#findThresholdMin(frame_full)
    
    #init du tracker
    tracker = MultipleObjectTracker()
    if args['magic'] == "1":
        mosaic = GenMosaic()
    
    count = 0
    hog = cv2.HOGDescriptor()
   
    while fvs.more():
        print('***************************************************************')
        # Take each frame
        frame_pos,frame = fvs.read()
        height , width, channel = frame.shape
        #frame = cv2.resize(frame,(width/2, height/2), interpolation = cv2.INTER_LINEAR)
        # Convert BGR to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not init_once:
            
            mask = np.zeros_like(frame)
            img = np.zeros_like(frame)
            
            init_once = True
        
        
        ret,thresh2 = cv2.threshold(gray,minThreshold,255,cv2.THRESH_BINARY_INV)
        
        thresh2_dilate = cv2.dilate(thresh2,kernel,iterations = 1)
        thresh2_erode = cv2.erode(thresh2_dilate ,kernel,iterations = 1)
        
        thresh2_median  = cv2.medianBlur(thresh2_erode,5)
    
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
            
            if perimeter > const.MAX_PERIMETER_BLOB:
                print("perimetre max atteint -> bordure génante")
                continue
            
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if args['magic'] == "1" :
                mosaic.addImage(frame[y:y+h, x:x+w])
            if args['flagTraining'] == "1":
                cv2.imwrite(args['output'] + 'train_'+ str(count) + '.jpg',frame[y-10:y+h+10, x-10:x+w+10])
                count = count + 1
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
                    ret,label,center = cv2.kmeans(Z,n,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                    for num in range(n):
                        centroids.append(int(float(center[num][0]) + x))
                        centroids.append(int(float(center[num][1]) + y))
                        centroids.append(np.sum(label == num))
            
            frame_2  = cv2.drawContours(frame  ,[box],0,(0,255,0),1)   
            cv2.circle(frame_2,(centroids[0], centroids[1]), 3, (0,0,255), -1)
            cv2.circle(frame_2,(centroids[3], centroids[4]), 3, (0,0,255), -1)
            kp1, des1 = hog.detectAndCompute(frame[y-20:y+h+20, x-20:x+w+20],None)
            print(kp1)
            #kp, descr = orb.compute(frame, [(centroids[0], centroids[1])])
            #print(descr)
            
            
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
        
        print("time: ",str(frame_pos), " num frame : " + str(numFrame))
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                print("PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))        
                mask = cv2.line(mask, (int(track.old_plot[0][0]),int(track.old_plot[0][1])),(int(track.plot[0][0]),int(track.plot[0][1])), color[track.label].tolist(), 1)                
                cv2.putText(frame_2,str(track.label),(int(track.plot[0][0]),int(track.plot[0][1])), font, 0.4,(255,0,0),1,cv2.LINE_AA)
        cv2.putText(frame_2,'frame ' + str(numFrame),(10,20), font, 0.4,(255,0,0),1,cv2.LINE_AA)
        #cv2.putText(frame,'nb tracks ' + str(len(tracker.tracks)),(10,40), font, 0.4,(255,0,0),1,cv2.LINE_AA)
        if flag_hide_tracks==True:
            cv2.putText(frame_2,"fade out activate",(10,60), font, 0.4,(255,0,0),1,cv2.LINE_AA)

        if flag_init_record == True:
            cv2.putText(frame_2,"save video",(10,80), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            
        img = cv2.addWeighted(mask, 1, frame_2, 1, 0)
        cv2.addWeighted(mask, alpha_1, mask, 0.5, alpha_2 ,mask)
        
        if args['no_preview'] == 1:
            #im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img = cv2.resize(img,(width, height), interpolation = cv2.INTER_LINEAR)
            cv2.imshow('res',img)
        
        numFrame = numFrame + 1
        
        if flag_init_record == True:
            out.write(img)
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
            elif k==ord('s'):
                if flag_init_record == False:
                    (h__, w__) = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    fd_video_out = args["output"] + "flyTracker_out.avi"
                    out = cv2.VideoWriter(fd_video_out,fourcc , 30, (w__,h__))
                    flag_init_record = True
                else:
                    flag_init_record = False
                    out.release()
                    
            else:
                print(k) # else print its value
    
    fvs.stop()


if __name__ == '__main__':
    __version__ = 0.3
    print("MouseTracker version  :" + str(__version__))
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-video", type=str, default=-1,
                    help="# relative path of the input video to analyse")
    ap.add_argument("-n", "--no-preview", type=str, default=1,
                    help="# desactivate the preview of the results")
    #not used
    ap.add_argument("-o", "--output", type=str, default=const.DIR_WORKSPACE,
                    help="# output directory")
    ap.add_argument("-e", "--flagTraining", type=str, default=0,
                    help="# training image extraction ")                
    ap.add_argument("-m", "--magic", type=str, default=0,
                    help="# magic option")
                    
    args = vars(ap.parse_args())
    
    if args["input_video"] == -1:
        args["input_video"] = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("vid","*.h264"),("all files","*.*")))
        if len(args["input_video"])==0:
            print("no input file -> no analysis")
            exit(0)
    main(args)
