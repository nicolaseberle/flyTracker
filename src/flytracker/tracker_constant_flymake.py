#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:37:21 2017

@author: neberle
"""

# TODO: add to cfg
MAX_NUM_MISSES_TRACK = 200
MAX_PIXELS_DIST_TRACK = 80  # TODO: use a ratio of bboxes instead (smaller bb means further away means shorter dist)
MAX_PIXELS_DIST_TRACK_START = int(MAX_PIXELS_DIST_TRACK/1.5)  # TODO: use a ratio of bboxes instead (smaller bb means further away means shorter dist)
MIN_IOU_TRACK = 0.5
MIN_IOU_TRACK_START = 0.5
MAX_PERIMETER_BLOB = 200
SURFACE_MAX = 60
DIAMETER_ROUND_BOX_CM = 8.6
DIAMETER_ROUND_BOX_PIXEL = 205
CONV_PX2CM = DIAMETER_ROUND_BOX_CM/DIAMETER_ROUND_BOX_PIXEL
VERBOSE_LIGHT = True
VERBOSE_FULL = False
NBFRAME_BEFORE_RECORDING = 30
DIR_WORKSPACE = '/home/pi/workspace/flyTracker/output'#A mettre en absolu
