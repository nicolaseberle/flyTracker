#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:09:54 2018

@author: neberle
"""
import cv2
import numpy as np

frame = np.zeros((200,200,3),dtype=np.uint8)
#noise = 0.5 * np.random.randn(200,200,3)+10
#noise = noise.astype(np.uint8)
#frame = np.array(frame + noise)

ratio = 0.08 * np.random.randn(2) + 0.5
majorRadius = 1.2 * np.random.randn(2) + 13
minorRadius = majorRadius*ratio

cv2.ellipse(frame,((100,100),(minorRadius[0], majorRadius[0]),0), (255,255,255) , -1,cv2.FILLED)
cv2.ellipse(frame,((109,100),(minorRadius[0], majorRadius[1]),90), (255,255,255) , -1,cv2.FILLED)
cv2.ellipse(frame,((94,98),(minorRadius[0], majorRadius[1]),80), (255,255,255) , -1,cv2.FILLED)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret,thresh2 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

converge = False
#easy function
liste_finale_ellipse = []
for cnt in contours:
    
    while converge == False:
        
        liste_score = []
        liste_ellipse = []
        for offset in range(0,100):
            
            if len(cnt[:][offset:offset+12])>=8:
                ellipse = cv2.fitEllipse(cnt[:][offset:offset+12])
                if ellipse[1][0]/ellipse[1][1]<0.65 and ellipse[1][0]/ellipse[1][1]>0.35 :
                    #cv2.ellipse(draw_frame, ellipse, (0,255,0), 1,cv2.LINE_AA)
                    pts = cv2.ellipse2Poly((int(ellipse[0][0]),int(ellipse[0][1])),(int(ellipse[1][0]), int(ellipse[1][1])), int(ellipse[2]), 0, 360,30);
                    score = 0
                    cn_to_delete = []
                    area = cv2.contourArea(pts)
                    #perimeter = cv2.arcLength(pts,True)
                    print(area)
                    #on exclut les ellipses trop petites
                    if area < 120:
                        continue
                    
                    cnt_to_delete = []
                    previous_cn=False
                    for pt in pts:    
                        for cn in cnt:
                            dist = np.linalg.norm(pt-cn)
                            if dist <= 2:
                                if previous_cn==True:
                                    consecutif = 0.5
                                else:
                                    consecutif = 0
                                previous_cn=True            
                            else:
                                consecutif = 0
                                previous_cn=False
                            score = score + 0.5/(dist+1) + consecutif
                             #   cnt_to_delete.append(cn)
                                
                    liste_score.append([score/len(pts),offset,area])
                    liste_ellipse.append(ellipse)
                    
                    #cv2.imshow("ellispe",draw_frame)
                    #cv2.waitKey(500)
        print(liste_score)
        ind_sorted = np.argsort(liste_score,0)
        best_offset = liste_score[ind_sorted[-1][0]][1]
        best_score = liste_score[ind_sorted[-1][0]][0]
        #print(liste_ellipse)
        print(best_score,best_offset)
        liste_ellipse = np.array(liste_ellipse)
        
        
        liste_finale_ellipse.append(liste_ellipse[ind_sorted[-1][0]])
        
        new_cnt = np.concatenate((cnt[:][:best_offset],cnt[:][best_offset+10:]),axis=0)
        #print(new_cnt)
        cnt = new_cnt
        if len(cnt[:][:])<=12:
            converge = True
            
liste_finale_ellipse = np.array(liste_finale_ellipse)
liste_finale_ellipse = tuple(map(tuple, liste_finale_ellipse))
#print(liste_finale_ellipse)
res_frame = np.copy(frame)
for elli in liste_finale_ellipse:
    #print(elli)
    cv2.ellipse(res_frame , elli, (0,255,0), 1,cv2.LINE_AA)
    
cv2.imshow("ellispe",res_frame)
cv2.waitKey(0)