

import cv2
import numpy as np
from multipleObjectTracker_3 import MultipleObjectTracker
import tracker_constant as const
import argparse


def main(args):
    #X vers le bas, Y vers la droite
    A = [ 45 , 95   ]
    B = [ 250 , 95  ]
    
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    color = np.random.randint(0,255,(255,3))
    
    lk_params = dict( winSize  = (20,20),
                       maxLevel = 1,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    

    ############################################################################
    cap = cv2.VideoCapture(args['input_video'])
    marge_x = 1
    marge_y = 20
    seuil_bas=70
    ############################################################################    
    
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    init_once = False
    numFrame = 0
    #init du tracker
    tracker = MultipleObjectTracker()

    while(cap.isOpened()):
        print('***************************************************************')
        # Take each frame
        ret, frame_full = cap.read()

        if ret == False:
            continue

        height, width, _ = frame_full.shape
        frame = frame_full[marge_y:height-marge_y, marge_x:width-marge_x]
        
        # Convert BGR to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.line(frame, (A[0],A[1]),(B[0],B[1]), np.array((0,255.,0)) , 2)
        cv2.imshow('frameRGB',frame)
        # initialization of mask and store frame    
        if not init_once:
            
            mask = np.zeros_like(frame)
            mask2 = np.zeros_like(frame)
            old_gray = np.zeros_like(frame)
            num_plot = 0
            init_once = True
        
        ret,thresh2 = cv2.threshold(gray,seuil_bas,255,cv2.THRESH_BINARY_INV)
        thresh2_dilate = cv2.dilate(thresh2,kernel,iterations = 1)
        thresh2_median  = cv2.medianBlur(thresh2_dilate,5)
        
        cv2.imshow('maskMedian',thresh2_median)
        im2, contours, hierarchy = cv2.findContours(thresh2_median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        pos_t = np.empty((0,1,4), dtype='float32')
        
        #For each blob/contour 
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            #if the perimeteter is too high, then this blob is ignored 
            if perimeter > const.MAX_PERIMETER_BLOB:
                continue
            #else this blob is a plot
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(area , ' ' , rect)
            if(area>80):
                (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                ellipse = cv2.fitEllipse(cnt)
                frame = cv2.ellipse(frame,ellipse, (0,255,0), 1)
            else:
                if(M['m00']!=0):
                    cx = (M['m10']/M['m00'])
                    cy = (M['m01']/M['m00'])
                    MA = 2
                    ma = 2
                    angle = 0
                else:
                    cx = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4)
                    cy = ((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)
                    angle = 0
                    MA = 2
                    ma = 2
                    angle = 0
            
            #cv2.drawContours(frame,[box],0,(0,0,255),2)
            current_plot = np.array([[[ cx,cy,angle*np.pi/180,area ]]],dtype='float32')
            #pos_t centroids of blob at time t
            pos_t = np.concatenate((pos_t,current_plot))
        #update of the tracker with new plots
        tracker.update_tracks(pos_t)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                print("PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))
                #mask = cv2.circle(mask,(track.plot[0][0],track.plot[0][1]), 2, color[track.label].tolist(), -1)
                mask = cv2.line(mask, (int(track.old_plot[0][0]),int(track.old_plot[0][1])),(int(track.plot[0][0]),int(track.plot[0][1])), color[track.label].tolist(), 2)
                #cv2.putText(frame,str(track.label),(track.plot[0][0],track.plot[0][1]), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            #if track.has_match is False:
                #cv2.putText(frame,str(track.label),(track.plot[0][0],track.plot[0][1]), font, 0.4,(48, 214, 232),1,cv2.LINE_AA)
        img = cv2.add(frame,mask)
        
        cv2.imshow('res',img)
        
        old_gray = gray.copy()
        
        numFrame = numFrame + 1
        pos_tm1=pos_t
        
        
        
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
    cap.release()


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-video", type=str, default='./dataset/out.mp4',
                    help="# relative path of the input video to analyse")
    
    args = vars(ap.parse_args())

    main(args)
