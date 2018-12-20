#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:36:44 2018

@author: neberle
"""

from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import argparse
from filevideostream import FileVideoStream
import cv2
import imutils
from pyzbar.pyzbar import ZBarSymbol
import logging

class AreneDetector(object):
    def __init__(self,pattern,image):
        self.pattern = pattern
        self.im = image
        self.arrArene = None
        self.nbArene = len(self.pattern)
    def computeArenePos(self):
        points = np.zeros((4,2))
        for dict in self.pattern:
             print(dict['pos'].mean(axis=0))
             numArene = np.int(dict['data'])-1
             points[numArene,:] = dict['pos'].mean(axis=0).transpose()

        left_x,top_y = np.min(points,axis=0)
        left_x = np.int(left_x)
        top_y = np.int(top_y)
        right_x,bottom_y = np.max(points,axis=0)
        right_x = np.int(right_x)
        bottom_y = np.int(bottom_y)
        width_arene = np.int(np.abs(right_x-left_x)/2)
        height_arene = np.int(np.abs(bottom_y-top_y)/2)
        posArene = []
        for dict in self.pattern:
            if dict['data'] == "1":
                dict['ArenePos'] = np.array([left_x,top_y,width_arene,height_arene ])#x,y,w,h
                posArene.append([1,(left_x,top_y),(left_x+width_arene,top_y+height_arene)])
                logging.info('Arene 1 x:%d y:%d w %d h:%d',
                    left_x,
                    top_y,
                    width_arene,
                    height_arene
                )
            elif dict['data'] == "2":
                dict['ArenePos'] = np.array([left_x+width_arene,top_y,width_arene,height_arene ])#x,y,w,h
                posArene.append([2,(left_x+width_arene,top_y),(left_x+2*width_arene,top_y+height_arene)])
                logging.info('Arene 2 x:%d y:%d w %d h:%d',
                    left_x+width_arene,
                    top_y,
                    width_arene,
                    height_arene
                )
            elif dict['data'] == "3":
                dict['ArenePos'] = np.array([left_x,top_y+height_arene ,width_arene,height_arene ])#x,y,w,h
                posArene.append([3,(left_x,top_y+height_arene),(left_x+width_arene,top_y+2*height_arene)])
                logging.info('Arene 3 x:%d y:%d w %d h:%d',
                    left_x,
                    top_y+height_arene,
                    width_arene,
                    height_arene
                )
            elif dict['data'] == "4":
                dict['ArenePos'] = np.array([left_x+width_arene,top_y+height_arene ,width_arene,height_arene ])#x,y,w,h
                posArene.append([4,(left_x+width_arene,top_y+height_arene),(left_x+2*width_arene,top_y+2*height_arene)])
                logging.info('Arene 4 x:%d y:%d w %d h:%d',
                    left_x+width_arene,
                    top_y+height_arene,
                    width_arene,
                    height_arene
                )
        return posArene

    def display(self):
        for dict in self.pattern:
           cv2.rectangle(self.im,(dict['ArenePos'][0],dict['ArenePos'][1]),(dict['ArenePos'][0]+dict['ArenePos'][2],dict['ArenePos'][1]+dict['ArenePos'][3])  , (0,255,0), 3)
        cv2.imshow("Results", self.im);
        cv2.waitKey(1000);
        cv2.destroyWindow("Results");

class QRCodeDetector(object):
    def __init__(self,image):
        self.pattern = [
              {"name": "QRcode_1","data":"1", "flag": "0","pos":[],"ArenePos":[]},
              {"name": "QRcode_2","data":"2", "flag": "0","pos":[],"ArenePos":[]},
              {"name": "QRcode_3","data":"3", "flag": "0","pos":[],"ArenePos":[]},
              {"name": "QRcode_4","data":"4", "flag": "0","pos":[],"ArenePos":[]}
            ]
        self.im = image
        self.decodedObjects = None


    def findAndReplace(self, obj):
        for dict in self.pattern:
            if dict['data'] == np.str(np.int8(obj.data)):
                if dict['flag'] == "0":
                    dict['flag'] = "1"
                    dict['pos'] = np.array([point for point in obj.polygon], dtype=np.float32)
                    return dict['name']

    def checkAllQRcodeDetected(self):
        err = 0
        for dict in self.pattern:
            if dict['flag'] == "0":
                err = 3001 #erreur 3001 : all QRCode are not detected

        return err
    def getQRCode(self,num):
        for dict in self.pattern:
            if dict['data'] == np.str(num) and dict['flag'] == "1":
                return dict['pos']

    def getPattern(self):
        return self.pattern

    def decode(self,thresh2) :
        # Find barcodes and QR codes
        self.decodedObjects = pyzbar.decode(thresh2, symbols=[ZBarSymbol.QRCODE],scan_locations=True)
        print(self.decodedObjects)
        # Print results
        for obj in self.decodedObjects:
            self.findAndReplace(obj)


    def scan(self):
        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        for threshold_i in range(20,140,2):
            ret,thresh2 = cv2.threshold(gray,threshold_i,255,cv2.THRESH_BINARY)

            # thresholded = cv2.inRange(gray,(0,0,0),(200,200,200))
            thresh2 = cv2.cvtColor(thresh2,cv2.COLOR_GRAY2BGR) # black-in-white

            self.decode(thresh2)

        return self.checkAllQRcodeDetected()

    # Display barcode and QR code location
    def display(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Loop over all decoded objects
        for num in range(1,5):
            # If the points do not form a quad, find convex hull
            points = self.getQRCode(num)

            #if len(points) > 4 :
            hull = cv2.convexHull(points)
            hull = list(map(tuple, np.squeeze(hull)))
            #else :
            #    hull = points;

            # Number of points in the convex hull
            n = len(hull)

            # Draw the convext hull
            for j in range(0,n):
                cv2.line(self.im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

            cv2.putText(self.im,str(num),hull[0], font, 1,(255,0,0),1,cv2.LINE_AA)
        # Display results
        cv2.imshow("Results", self.im);
        cv2.waitKey(0);


# Main
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--input-video", type=str, default=-1,
                    help="# relative path of the input video to analyse")
    ap.add_argument("-i", "--input-image", type=str, default=-1,
                    help="# relative path of the input video to analyse")
    args = vars(ap.parse_args())

    if args['input_video']!=-1:
        fvs = FileVideoStream(args['input_video'],1).start()
        frame_pos,frame = fvs.read()
        fvs.more()
        frame_pos,frame = fvs.read()
        fvs.more()
        frame_pos,frame = fvs.read()
    if args['input_image']!=-1:
        frame = cv2.imread(args['input_image'], flags=cv2.IMREAD_COLOR)

    print(frame.shape)
    cv2.imshow("input", frame )
    cv2.waitKey(0)

    # frame = imutils.resize(frame, width=1000)
    # frame = imutils.resize(frame, width=400)

    QRCode = QRCodeDetector(frame)
    err = QRCode.scan()
    print(QRCode.getPattern())
    QRCode.display()
    if err > 0:
        print("erreur detection arene")
        exit(0)

    Arene = AreneDetector(QRCode.getPattern(),frame)
    Arene.computeArenePos()
    Arene.display()
