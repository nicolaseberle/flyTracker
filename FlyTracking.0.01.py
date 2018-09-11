

import cv2
import numpy as np
from multipleObjectTracker_3 import MultipleObjectTracker
import tracker_constant as const
import argparse
from matplotlib import pyplot as plt

def findCentroids(ext_img):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    size = np.size(ext_img)
    skel = np.zeros(ext_img.shape,np.uint8)
    
    iteration = 0
    iteration_lim = 100
    while( not done or iteration<iteration_lim):
        eroded = cv2.erode(ext_img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(ext_img,temp)
        skel = cv2.bitwise_or(skel,temp)
        ext_img = eroded.copy()
    
        zeros = size - cv2.countNonZero(ext_img)
        if zeros==size:
            done = True
        iteration = iteration + 1
        if iteration > iteration_lim:
            break
    nb_element = cv2.sumElems(skel)
    nb_element = float(nb_element[0]) / 255.0
    
    return nb_element

def main(args):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    flag_hidden_tracks = False
    
    alpha_1 = 0.5 # transparency of the mask to see the tracks
    alpha_2 = 0 # transparency of the mask to see the tracks
    #X vers le bas, Y vers la droite
    
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    color = np.random.randint(0,255,(255,3))
    
    lk_params = dict( winSize  = (20,20),
                       maxLevel = 1,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    

    ############################################################################
    cap = cv2.VideoCapture(args['input_video'])
    marge_x = 1
    marge_y = 1
    seuil_bas=60
    ############################################################################    
    
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    for counter in range(30):
        ret, frame_full = cap.read()
        
        
    init_once = False
    numFrame = 30
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
        if args['no_preview'] == 1:
            cv2.imshow('frameRGB',frame)
        # initialization of mask and store frame    
        if not init_once:
            
            mask = np.zeros_like(frame)
            old_gray = np.zeros_like(frame)
            init_once = True
        
        ret,thresh2 = cv2.threshold(gray,seuil_bas,255,cv2.THRESH_BINARY_INV)
        #thresh2_dilate = cv2.erode(thresh2,kernel,iterations = 1)
        thresh2_dilate = cv2.dilate(thresh2,kernel,iterations = 1)
        thresh2_median  = cv2.medianBlur(thresh2_dilate,5)
        
        if args['no_preview'] == 1:
            cv2.imshow('maskMedian',thresh2_median)
            
        im2, contours, hierarchy = cv2.findContours(thresh2_median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        pos_t = np.empty((0,1,4), dtype='float32')
        
        #For each blob/contour 
        for cnt in contours:
            
            rect = cv2.minAreaRect(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt,True)
            #if the perimeteter is too high, then this blob is ignored 
            if perimeter > const.MAX_PERIMETER_BLOB:
                continue
            #else this blob is a plot
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # define criteria and apply kmeans()
            
            
            nb_element = findCentroids(thresh2_median[y:y+h, x:x+w])
            #warning : two plots are merged
            if nb_element >= 15 and nb_element<31  and area > 60 and area <180:
                
                pixelpointsCV2 = cv2.findNonZero(thresh2_median[y:y+h, x:x+w])
                Z = np.float32(pixelpointsCV2)
                ret,label,center = cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                dist_center = np.sqrt(float(center[0][0]-center[1][0])*float(center[0][0]-center[1][0])+float(center[0][1]-center[1][1])*float(center[0][1]-center[1][1]))
                angle = 0
                MA = 2
                ma = 2
                angle = 0
                #check that is not the same blob
                if dist_center < 1:
                    #it's the same blob. limit the total element with one element
                    max_element = 1
                else:
                    #it's not the same blob. 2 elements
                    max_element = 2
                    
                for num in range(max_element):
                    cx = float(center[num][0]) + x
                    cy = float(center[num][1]) + y
                    area = np.sum(label == num)
                    #we record the first blob, the second will be at the and of the loop
                    current_plot = np.array([[[ cx,cy,angle*np.pi/180,0*area/10 ]]],dtype='float32')
                    #pos_t centroids of blob 1 at time t
                    pos_t = np.concatenate((pos_t,current_plot))

                        
                
                
             #warning : two plots are merged
            elif nb_element >=31  and area >= 180:
                
                pixelpointsCV2 = cv2.findNonZero(thresh2_median[y:y+h, x:x+w])
                Z = np.float32(pixelpointsCV2)
                ret,label,center = cv2.kmeans(Z,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                
                
                #check that is not the same blob
                angle = 0
                MA = 2
                ma = 2
                angle = 0
                
                for num in range(2):
                    cx = float(center[num][0]) + x
                    cy = float(center[num][1]) + y
                    area = np.sum(label == num)
                    #we record the first blob, the second will be at the and of the loop
                    current_plot = np.array([[[ cx,cy,angle*np.pi/180,0*area/10 ]]],dtype='float32')
                    #pos_t centroids of blob 1 at time t
                    pos_t = np.concatenate((pos_t,current_plot))
#               # Now separate the data, Note the flatten()
#                A = Z[label==0]
#                B = Z[label==1]
#                C = Z[label==2]
#                # Plot the data
#                plt.scatter(A[:,0],A[:,1])
#                plt.scatter(B[:,0],B[:,1],c = 'r')
#                plt.scatter(C[:,0],C[:,1],c = 'g')
#                plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
#                plt.xlabel('Height'),plt.ylabel('Weight')
#                plt.show()     
            else:
                
                if(area>10000):
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
                current_plot = np.array([[[ cx,cy,angle*np.pi/180,0*area/10 ]]],dtype='float32')
                #pos_t centroids of blob at time t
                pos_t = np.concatenate((pos_t,current_plot))
        #update of the tracker with new plots
        tracker.update_tracks(pos_t)
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("num frame : " + str(numFrame))
        for i, track in enumerate(tracker.tracks):
            if track.has_match is True:
                print("PISTE "  + str(i) + " label : " + str(track.label)  + " X: " + str(track.plot[0][0]) + " Y: " + str(track.plot[0][1]) + " Theta: " + str(track.plot[0][2]) + " S: " + str(track.plot[0][3]))
                #mask = cv2.circle(mask,(track.plot[0][0],track.plot[0][1]), 2, color[track.label].tolist(), -1)
                mask = cv2.line(mask, (int(track.old_plot[0][0]),int(track.old_plot[0][1])),(int(track.plot[0][0]),int(track.plot[0][1])), color[track.label].tolist(), 1)
                cv2.putText(frame,str(track.label),(track.plot[0][0],track.plot[0][1]), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            #if track.has_match is False:
                #cv2.putText(frame,str(track.label),(track.plot[0][0],track.plot[0][1]), font, 0.4,(48, 214, 232),1,cv2.LINE_AA)
        cv2.putText(frame,'frame ' + str(numFrame),(10,20), font, 0.4,(255,0,0),1,cv2.LINE_AA)
        
        if flag_hidden_tracks==True:
            cv2.putText(frame,"fade out activate",(10,40), font, 0.4,(255,0,0),1,cv2.LINE_AA)
            
        img = cv2.addWeighted(mask, 1, frame, 1, 0)
        cv2.addWeighted(mask, alpha_1, mask, 0.5, alpha_2 ,mask)
        
        if args['no_preview'] == 1:
            cv2.imshow('res',img)
        
        old_gray = gray.copy()
        
        numFrame = numFrame + 1
        pos_tm1 = pos_t
        
        
        if args['no_preview'] == 1:
            k = cv2.waitKey(25)
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k==ord('a'):
                if flag_hidden_tracks == False:
                    alpha_1 = 0.495
                    alpha_2 = -1
                    flag_hidden_tracks = True
                    continue
                else:
                    alpha_1 = 0.5
                    alpha_2 = 0
                    flag_hidden_tracks = False
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            elif k==ord('p'):
                while True:
                    c = cv2.waitKey(25)        
                    if c == ord('p'):
                        break
            else:
                print(k) # else print its value
        
    cap.release()


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-video", type=str, default='/media/neberle/6CEC5E62EC5E271C/Backup_Linux/fly_tracker/dataset_/test.avi',
                    help="# relative path of the input video to analyse")
    ap.add_argument("-n", "--no-preview", type=str, default=1,
                    help="# desactivate the preview of the results")
    #not used
    ap.add_argument("-o", "--output", type=str, default=const.DIR_WORKSPACE,
                    help="# output directory")
    
    args = vars(ap.parse_args())

    main(args)
