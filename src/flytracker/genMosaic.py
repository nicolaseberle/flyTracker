#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:14:55 2018

@author: neberle
"""

import itertools

import cv2
import numpy as np

class GenMosaic(object):
    def __init__(self):
        self.w = 20
        self.h = 30
        self.img_width = 20
        self.img_height = 20
        self.margin=0
        self.n = self.w*self.h
        self.nbImInStack = 0
        self.stack = []
    def addImage(self,img):
        resized_image = cv2.resize(img, (self.img_width, self.img_height)) 
        self.stack.append(resized_image)
        self.nbImInStack = self.nbImInStack + 1
        if self.nbImInStack == self.n:
            self.displayMosaic()
            self.nbImInStack = 0
            self.stack = []
    
    def displayMosaic(self):
        img_h, img_w, img_c = self.stack[0].shape
        
        imgmatrix = np.zeros((img_h * self.h,
                              img_w * self.w,
                              3),
                            np.uint8)
        imgmatrix.fill(255)    

        positions = itertools.product(range(self.w), range(self.h))
        for (x_i, y_i), img in zip(positions, self.stack):
            x = x_i * (img_w )
            y = y_i * (img_h )
            imgmatrix[y:y+img_h, x:x+img_w, :] = img
        cv2.imshow('mosaic',imgmatrix)
        